import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# API 키 입력 또는 환경 변수에 직접 설정해야함.
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "(api_key 입력)"

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 이미지 URL 포함 메시지
message_url = HumanMessage(
    content=[
        {
            "type": "text",
            "text": (
                "URL 이미지 안에 있는 사람의 체형을 분석해줘. "
                "{역삼각형, 직사각형, 둥근형, 삼각형, 모래시계형} 중 하나로 구분하고, "
                "기준은 다음과 같아. "
                "역삼각형(남성형): 어깨 > 가슴 > 허리 > 엉덩이, "
                "직사각형(직선형): 어깨 ≈ 가슴 ≈ 허리 ≈ 엉덩이, "
                "둥근형(원형): 허리 ≥ 어깨 ≈ 엉덩이, "
                "삼각형(배형): 엉덩이 > 허리 > 가슴 > 어깨, "
                "모래시계형(균형형): 가슴 ≈ 엉덩이 > 허리"
            )
        },
        {
            "type": "image_url",
            "image_url": "https://sports.hankooki.com/news/photo/202008/img_6532284_0.jpg"
            # 사용자의 이미지가 들어갈 수 있도록 수정.
        },
    ]
)


result_url = llm.invoke([message_url])

print("Response for URL image:")
print(result_url.content if isinstance(result_url.content, str) else str(result_url.content))
