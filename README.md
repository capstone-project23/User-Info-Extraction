# User-Info-Extraction
유저 정보 추출 구현

### 필요 라이브러리
- 퍼스널 컬러 예측
```
import os
import cv2
import numpy as np
import joblib
import dlib
from tensorflow.keras.models import load_model
```

- 체형 분석 위한 llm
```
import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
```
