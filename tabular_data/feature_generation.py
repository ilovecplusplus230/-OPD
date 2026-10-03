"""本地 reasoned 生成器的客户端标记；生成算法见 feature_proposals.py。"""

class OfflineFeatureClient:
    """不请求外部模型；记录本地生成轮数，避免误报为实际 LLM 调用。"""

    def __init__(self):
        self.real_call_count = 0
        self.mock_fallback_count = 0
