from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import UnstructuredURLLoader
from ssl_helper import init
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

from pydantic import BaseModel, Field

load_dotenv()

# 初始化ssl
init()


class Line(BaseModel):
    character: str = Field(description="角色")
    content: str = Field(description="台词的具体内容，其中不包含角色的名字")


class XianSheng(BaseModel):
    script: list[Line] = Field(description="一段相声的台词")


def url_get_news(url: str):
    urls = [
        url
    ]

    loader = UnstructuredURLLoader(urls=urls)

    text_splitter = RecursiveCharacterTextSplitter(separators=["正文", "责任编辑"], chunk_size=100, chunk_overlap=20,
                                                   length_function=len)

    data = loader.load_and_split(text_splitter)

    return data[1:2]


def get_news(news) -> XianSheng:
    llm = OpenAI(max_tokens=1500)

    prompt_template = """总结这段新闻的内容:
    
            “{text}”
            
            总结:"""

    chinese_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    chain = load_summarize_chain(llm, chain_type="stuff", prompt=chinese_prompt)

    summary = chain.run(news)

    openai_chat = ChatOpenAI(model_name="gpt-3.5-turbo")

    template = """\
        我将给你一段新闻的概括，请按照要求把这段新闻改写成小狐与陈总的相声段子。
    
        新闻：{新闻}
        要求：{要求}
        {output_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=XianSheng)

    prompt = PromptTemplate(
        template=template,
        input_variables=["新闻", "要求"],
        partial_variables={"output_instructions": parser.get_format_instructions()},
    )

    msg = [HumanMessage(
        content=prompt.format(新闻=summary, 要求="风趣幽默，十分讽刺，剧本对话角色为郭德纲与于谦，以他们的自我介绍开始"))]

    res = openai_chat(msg)

    x_s = parser.parse(res.content)
    return x_s


def get_xs(url: str):
    doc = url_get_news(url)

    xs = get_news(doc)
    return xs


# url_ = "https://news.sina.com.cn/gov/xlxw/2024-02-05/doc-inafxzvk7911699.shtml"
# result = get_xs(url_)
#
# print(f"返回最终结果：{result}")
