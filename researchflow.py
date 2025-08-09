from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from pydantic import BaseModel
from typing import List
from crewai.tools import tool, BaseTool
from crewai.flow.flow import Flow, listen, router, start
from pydantic import Field
from typing import Type
import io
import os
from langchain_community.utilities import GoogleSerperAPIWrapper

os.environ["SERPER_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["OTEL_SDK_DISABLED"] = "true"

#geminillm = LLM(model="gemini/gemini-1.5-flash-8b")
geminillm = LLM(model="gemini/gemini-2.0-flash")
llm = LLM(model="ollama/llama3.2:latest", base_url="http://localhost:11434")

#serper_tool = SerperDevTool(search_url="https://google.serper.dev/scholar", n_results=5)
#serper_tool = SerperDevTool()

search = GoogleSerperAPIWrapper()

class MyToolInput(BaseModel):
    """Input schema for SearchTool."""
    person: str = Field(..., description="Person to be searched on Internet.")
    company: str = Field(..., description="Name of the company the person is associated with.")

class SearchTool(BaseTool):
    name: str = "Search Tool"
    description: str = "Useful for search-based queries. Use this to find current information about a person."
    args_schema: Type[BaseModel] = MyToolInput
    search: GoogleSerperAPIWrapper = Field(default_factory=GoogleSerperAPIWrapper)

    def _run(self, person: str, company:str) -> str:
        """Execute the search query and return results"""
        print("Search query 1 -> ", person)
        print("Search query 2 -> ", company)
        try:
            return self.search.run(person + " " + company)
        except Exception as e:
            return f"Error performing search: {str(e)}"

# Person name model
class PersonName(BaseModel):
    name: List[str]

# Research State - storing list of names and their corresponding details
class ResearchState(BaseModel):
    intent: str = ""
    user_query: str = ""
    names: List[str] = []
    details: List[str] = []
 
class ResearchFlow(Flow[ResearchState]):

    @start()
    def start_conversation(self):
        print("Starting Research Flow \n")
        print("Tell me what do you want to do?\n")
        query = input()
        print("You have entered -> ", query)

        # Define Intent Extraction Agent
        intentextractor_agent = Agent(
            role="Intent extractor",
            goal="""
            You have to extract user's intent from given text {text}
            If the user wants to know about a company, just return "company"
            If the user wants to know the details about some people in a company, just return "people"
            Just return a single word that describes the intent. Do not return a long phrase or a sentence.
            """,
            backstory="You are an expert in finding user's intent from a given text. You return the user's intent as a single word.",
            llm=llm
            )

        # Define Intent Extraction Task
        intentextraction_task = Task(
            description="""
            Find out user's intent from the given text {text}
            """,
            expected_output="""
            A single word that describes the intent.
            """,
            agent=intentextractor_agent
            )

        intent_extraction_crew=Crew(
            agents=[intentextractor_agent],
            tasks=[intentextraction_task],
            process=Process.sequential,
            verbose=True
            )

        # Start the Intent Extraction flow
        result = intent_extraction_crew.kickoff(inputs={'text':query})
        print("User's Intent -> ", result.raw) 
        self.state.intent=result.raw
        self.state.user_query=query
    
    @router(start_conversation)
    def conversation_router(self):
        print("Inside conversation router -> Intent is ", self.state.intent)
        if "people" in self.state.intent.strip().lower():
            print("******** Looking for person details *********")
            return "people"
        if "company" in self.state.intent.strip().lower():
            print("******** Looking for company details *********")
            return "company"

    @listen("company")
    def research_company(self):
        print("OK, you want to research about a company")
        print("You have already entered ", self.state.user_query)

        # Define Company Name Extraction Agent
        companyname_extractor_agent = Agent(
            role="Company Name extractor",
            goal="""
            You have to extract the name of a company from given text {text}
            Just return the name of the company. Do not return anything else or any other text.
            """,
            backstory="You are an expert in finding names of company from a given text",
            llm=llm
            )

        # Define Company Name Extraction Task
        company_name_extraction_task = Task(
            description="""
            Find out name of company from the given text {text}
            """,
            expected_output="""
            Just the name of company
            """,
            agent=companyname_extractor_agent
            )

        # Define Company Leadership Research Agent
        company_leadership_research_agent = Agent(
            role="Company Leadership Researcher",
            goal="""
            You have to do research on all the top leaders of a company. 
            Find out details about all the top leaders of a company.
            Get their professional history, their current job profile and areas of interest
            """,
            backstory="You are an expert in doing research on top leaders of a company",
            llm=geminillm
            )

        # Define Company Leadership Research Task
        company_leadership_research_task = Task(
            description="""
            Perform detailed research on a all the top leaders of a company
            """,
            expected_output="""
            Summary report on all the top leaders of the company
            """,
            agent=company_leadership_research_agent
            )

        # Define Company Research Agent
        company_research_agent = Agent(
            role="Company Researcher",
            goal="""
            You have to do research on a company 
            """,
            backstory="You are an expert in doing research on a company",
            llm=geminillm
            )

        # Define Company Research Task
        company_research_task = Task(
            description="""
            Perform detailed research on a company 
            """,
            expected_output="""
            Summary report on the company
            """,
            agent=company_research_agent
            )

        # Define Crew
        company_name_extraction_crew=Crew(
            agents=[companyname_extractor_agent, company_research_agent],
            tasks=[company_name_extraction_task, company_research_task],
            process=Process.sequential,
            verbose=True
            )

        # Launch crew
        result = company_name_extraction_crew.kickoff(inputs={'text':self.state.user_query})
        print("Company Name -> ", result)

    @listen("people")
    def extract_names(self):
        print("OK, you want to know details about some people !! \n")
        print("Enter Company name, if any. If you do not know the company name, just press Enter\n")
        company = input()
        print("You have entered company -> ", company)
        print("Now, enter the text, followed by END word in a new line")
        print("If you want to use the same text entered earlier, just type END and press Enter")
        lines = []
        while True:
            line = input()
            if line == 'END':
                break
            lines.append(line)

        if not lines:
            lines.append(self.state.user_query)

        print("\nYou have entered -> ", lines)

        # Define Name Extraction Agent
        nameextractor_agent = Agent(
            role="Name extractor",
            goal="""
            You have to extract names of {type} from given text {text}
            """,
            backstory="You are an expert in finding names of {type} from a given text",
            llm=llm
            )

        # Define Name Extraction Task
        nameextraction_task = Task(
            description="""
            Find out names of {type} from the given text {text}
            """,
            expected_output="""
            List out all the names of {type}, one name in each line
            """,
            agent=nameextractor_agent,
            output_pydantic=PersonName
            )

        name_extraction_crew=Crew(
            agents=[nameextractor_agent],
            tasks=[nameextraction_task],
            process=Process.sequential,
            verbose=True
            )

        """
        # Let us find if there are any company names in the given text
        result = name_extraction_crew.kickoff(inputs={'type':"companies", 'text':lines})
        print("Result of Name Extraction Task for Companies -> ", result)
        companies = io.StringIO()
        for name in result.pydantic.name:
            print("Company Name is -> ", name)
            companies.write(name)
            companies.write(" ")
        print("List of Companies -> ", companies.getvalue())
        """

        @tool("Find details about a person")
        def get_person_details(person:str, company:str) -> str:
            """This tool finds details about a person by searching over the Internet and then returns the information. If the company name is known, use that information as well to search for related person."""
            print("******* Finding details about ", person)
            print("******* related to company ", company)
            search = GoogleSerperAPIWrapper()
            query = person + " " + company
            result = search.run(query)
            #result = serper_tool.run(search_query=query)
            print("Search result -> ", result)
            return result

        # Define Person Researcher Agent
        researcher_agent = Agent(
            role="Researcher",
            goal="""
            Find details about a person from Internet by using the provided tool. If the person's company name is also provided, pass on that information to the tool as well.
            Use the information returned by the tool to frame your response.
            Call the tool only once and use the output from the tool to generate response.
            If the tool output is in some other language than English, then translate the tool output to English and then generate response  
            """,
            backstory="""
            You are an expert in finding details about a person by searching various resources in Internet. 
            If the person's company name is provided, you use that information as well to search for that person in relation to that company.
            You use the provided tool to find information about the person and then present it in a concise fashion.
            Additional rules for Tools:
            -----------------
            1. Regarding the Action Input (the input to the action, just a simple python dictionary, enclosed
            in curly braces, using \" to wrap keys and values.)
            For example for the following schema:
            ```
            class MyToolInput(BaseModel):
                person: str = Field(..., description="Person to be searched on Internet.")
                company: str = Field(..., description="Name of the company the person is associated with.")
            ```
            Then the input should be a JSON object with
            - person: The Person to be searched on Internet
            - company: Name of the company the person is associated with
            """,
            tools=[SearchTool(result_as_answer=True)],
            llm=llm,
            max_iter=2,
            verbose=True
            )

        # Let us now find the names of people in the given text
        result = name_extraction_crew.kickoff(inputs={'type':"people", 'text':lines})
        print("Result of Name Extraction Task for People -> ", result)
        final_result = io.StringIO()
        for name in result.pydantic.name:
            print("Person's Name is -> ", name)
            # For each name, invoke Crew to get details about the person
            # For better result, add the company name also along with person's name

            # Define Research Task
            research_task = Task(
            description="""
            Find out details about {person} related to {company} by searching on Internet
            """,
            expected_output="""
            Find out details about {person} related to {company} provided as input to this task
            """,
            agent=researcher_agent,
            max_retries=2
            )
            person_details_crew=Crew(
            agents=[researcher_agent],
            tasks=[research_task],
            process=Process.sequential,
            verbose=True
            )
            modified_name = name + " " + company
            #result = person_details_crew.kickoff(inputs={'person':modified_name})
            #result = person_details_crew.kickoff(inputs={'person':name})
            result = person_details_crew.kickoff(inputs={'person':name, 'company':company})
            print("Result of Person Details Task -> ", result)
            self.state.names.append(name)
            self.state.details.append(research_task.output.raw)
            headerstring = "********************  Details about " + name + " **********************"
            final_result.write(headerstring)
            final_result.write("\n\n")
            final_result.write(research_task.output.raw)
            final_result.write("\n")
            final_result.write("\n")
            final_result.write("\n")
        # Final summary
        print("\n********* Summary of result **********\n")
        print("\n********* START **********\n")
        print(final_result.getvalue())
        print("\n\n********* END **********\n")

    @listen(extract_names)
    def generate_summary(self): 
        """
        print("List of names -> ", self.state.names)   
        print("List of details -> ", self.state.details)   
        """

        # Define Person Summarizer Agent
        summarizer_agent = Agent(
            role="Summarizer",
            goal="""
            Write a summary of details about every person provided as input to you. Write a separate paragraph for every person.
            """,
            backstory="""
            You are an expert in summarizing details of people provided as input to you.
            You create a separate paragraph for every person. 
            And in each paragraph, you write details about the person.
            """,
            llm=llm,
            verbose=True
            )

        # Define Summary Write Task
        summary_task = Task(
        description="""
        Write a summary of every {person} using corresponding details from {details}
        """,
        expected_output="""
        A separate paragraph with details about every person using provided input to this task
        """,
        agent=summarizer_agent,
        max_retries=2
        )

        summary_crew=Crew(
        agents=[summarizer_agent],
        tasks=[summary_task],
        process=Process.sequential,
        verbose=True
        )

        """
        result = summary_crew.kickoff(inputs={'person':self.state.names, 'details':self.state.details})
        print("Result of Summary Task -> ", result)
        """

# Launch Crew
def main():
    research_flow = ResearchFlow()
    research_flow.kickoff()

if __name__ == "__main__":
    main()
