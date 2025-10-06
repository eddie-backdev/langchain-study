# LangChain 기반 gRPC AI 서비스

이 프로젝트는 LangChain, Qdrant, OpenAI를 활용해 질문 추천과 혜택 추천을 제공하는 gRPC 마이크로서비스입니다.

## 필수 준비
- Python 3.9 이상 (권장 3.11)
- Qdrant 인스턴스 (로컬 또는 매니지드)
- OpenAI API Key

다음 명령으로 의존성을 설치하세요.

```bash
pip install -r requirements.txt
```

## 환경 변수
- `OPENAI_API_KEY`: OpenAI 인증 토큰
- `EMBEDDING_MODEL` (선택, 기본 `text-embedding-3-small`)
- `CHAT_MODEL` (선택, 기본 `gpt-4o-mini`)
- `QDRANT_HOST`, `QDRANT_PORT` (기본 `localhost`, `6333`)
- `QUESTION_COLLECTION`, `BENEFIT_COLLECTION` (선택, 기본 `questions`, `benefits`)
- `GRPC_PORT` (선택, 기본 `50051`)

## 실행 방법
```bash
python -m app.main
```

서버가 기동되면 gRPC 엔드포인트는 `ai.AiService` 네임스페이스로 노출됩니다.

## 주요 RPC
- `EmbedText`: 문장 배열을 받아 임베딩 벡터를 반환합니다.
- `RecommendQuestion`: 질문을 분류하고 추천 이유 및 관련 질문을 제공합니다.
- `RecommendBenefits`: 후보 혜택을 재정렬하고 사용자 맞춤 이유를 생성합니다.

proto 스키마는 `protos/ai_service.proto`에 있으며, 필요한 gRPC 스텁은 `app/generated`에 포함되어 있습니다.
