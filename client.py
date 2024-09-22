from llmonkey.llmonkey import LLMonkey

llmonkey = LLMonkey()

print("Available providers:", llmonkey.providers)
# print("Using OpenAI")

# response = llmonkey.generate_chat_response(
#     provider="openai",
#     model_name="gpt-3.5-turbo",
#     user_prompt="Hello! How are you?",
#     system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
# )

# print(response)

text = """
A wiki (/ˈwɪki/ ⓘ WI-kee) is a form of hypertext publication on the internet which is collaboratively edited and managed by its audience directly through a web browser. A typical wiki contains multiple pages that can either be edited by the public or limited to use within an organization for maintaining its internal knowledge base.

Wikis are powered by wiki software, also known as wiki engines. Being a form of content management system, these differ from other web-based systems such as blog software or static site generators in that the content is created without any defined owner or leader. Wikis have little inherent structure, allowing one to emerge according to the needs of the users.[1] Wiki engines usually allow content to be written using a lightweight markup language and sometimes edited with the help of a rich-text editor.[2] There are dozens of different wiki engines in use, both standalone and part of other software, such as bug tracking systems. Some wiki engines are free and open-source, whereas others are proprietary. Some permit control over different functions (levels of access); for example, editing rights may permit changing, adding, or removing material. Others may permit access without enforcing access control. Further rules may be imposed to organize content. In addition to hosting user-authored content, wikis allow those users to interact, hold discussions, and collaborate.[3]

There are hundreds of thousands of wikis in use, both public and private, including wikis functioning as knowledge management resources, note-taking tools, community websites, and intranets. Ward Cunningham, the developer of the first wiki software, WikiWikiWeb, originally described wiki as "the simplest online database that could possibly work".[4] "Wiki" (pronounced [wiki][note 1]) is a Hawaiian word meaning "quick".[5][6][7]

The online encyclopedia project Wikipedia is the most popular wiki-based website, as well being one of the internet's most popular websites, having been ranked consistently as such since at least 2007.[8] Wikipedia is not a single wiki but rather a collection of hundreds of wikis, with each one pertaining to a specific language. The English-language Wikipedia has the largest collection of articles, standing at 6,885,952 as of September 2024.[9]
"""

print("Using Groq")
for i in range(50):
    response = llmonkey.generate_chat_response(
        provider="groq",
        model_name="llama-3.1-70b-versatile",
        user_prompt=f"Hello! How are you? Please summarize following text: {text}",
        system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
        max_tokens=1000,
    )
    print(response)

# print("Using DeepInfra")
# response = llmonkey.generate_chat_response(
#     provider="deepinfra",
#     model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
#     user_prompt="Hello! How are you?",
#     system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
# )
# print(response)

# print("Using test")
# response = llmonkey.generate_chat_response(
#     provider="test",
#     model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
#     user_prompt="Hello! How are you?",
#     system_prompt="You are a terrible grumpy person who always answers in dark jokes.",
# )
# print(response)
