"""当前正式训练数据集：仅保留用户指定的三个任务。"""

DATASETS = ('jungle_chess', 'balance_scale', 'chess_krk')
RARE_CLASS_DATASETS = ("balance_scale",)
RELATIONAL_DATASETS = ("chess_krk",)
ROUND3_DATASETS = ()  # 原第三轮数据已按用户要求移除。

SPECS = {
    'jungle_chess': {
        "id": 41027,
        "slug": "jungle_chess_2pcs_raw_endgame_complete",
        "name": "Jungle Chess RAW",
        "task": "classification",
        "rows": 44819,
        "doi": None,
        "provider": "OpenML",
        "citation": "J. N. van Rijn and J. K. Vis. Endgame Analysis of Dou Shou Qi. ICGA Journal "
        "37(2):120–124, 2014.",
        "description": "从双方棋子的 rank、file、strength 六个原始字段预测三类残局结果；OpenML-CC18 基准。",
        "download_url": "https://openml.org/data/v1/download/18631418/jungle_chess_2pcs_raw_endgame_complete.arff",
        "md5": "25ca110c24c9dc3ffdcb5c2453382778",
        "target": "class",
        "features": 6,
        "classes": 3,
        "scope": "IID random split benchmark; not held-out-piece-pair generalization",
    },
    'balance_scale': {
        "id": 11, "name": "Balance Scale", "task": "classification",
        "rows": 625, "features": 4, "classes": 3, "doi": None,
        "provider": "OpenML", "target": "class",
        "citation": "Zhang et al. (2023). OpenFE: Automated Feature Generation with "
                    "Expert-level Performance. ICML, Table 7 (balance-scale).",
        "description": "左右两侧重量及距离四列，预测左倾/平衡/右倾；平衡类 49/625。",
        "download_url": "https://openml.org/data/v1/download/11/balance-scale.arff",
        "md5": "76938608d472f620c170cef9c8c1fa65",
        "original_source_url": "https://archive.ics.uci.edu/dataset/12/balance+scale",
        "benchmark_reference": "https://proceedings.mlr.press/v202/zhang23ay/zhang23ay.pdf#page=9",
        "literature_scope": "ICML 2023 OpenFE 自动特征生成的直接实验数据集。",
        "scope": "Small mechanistic benchmark (625 rows); minority training count is about 10. "
                 "Not evidence of large-scale real-world generalization.",
        "minority_tier": "relaxed_le_10_percent",
    },
    'chess_krk': {
        "id": 1481, "name": "Chess KRK", "task": "classification",
        "rows": 28056, "features": 6, "classes": 18, "doi": "10.24432/C57W2S",
        "provider": "OpenML", "target": "Class",
        "citation": "Fernández-Delgado et al. (2014). Do we Need Hundreds of Classifiers to Solve "
                    "Real World Classification Problems? JMLR, Table 1 (chess-krvk). "
                    "Bain, M. & Hoff, A. (1994). Chess (King-Rook vs. King). UCI.",
        "description": "白王、白车、黑王的六个坐标，保留和棋及 0–16 步取胜的 18 个原始类别。",
        "download_url": "https://openml.org/data/v1/download/1590570/kr-vs-k.arff",
        "md5": "c9a253e9da583db17676b756d39ade29",
        "original_source_url": "https://archive.ics.uci.edu/dataset/23/chess+king+rook+vs+king",
        "benchmark_reference": "https://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf#page=5",
        "literature_scope": "JMLR 分类基准；另有 FLAIRS 2012 的距离特征研究，不能称其为 AAAI 主会。",
        "additional_reference": "https://cdn.aaai.org/ocs/4430/4430-21453-1-PB.pdf",
        "scope": "IID endgame positions; 18 nominal labels are retained, not regression on encoded class IDs. "
                 "Rare classes have very few test cases; not independent-game/trajectory generalization.",
        "minority_tier": "strict_le_5_percent",
        "rename_columns": {"V1": "white_king_file", "V2": "white_king_rank", "V3": "white_rook_file",
                           "V4": "white_rook_rank", "V5": "black_king_file", "V6": "black_king_rank"},
        "coordinate_columns": ["white_king_file", "white_king_rank", "white_rook_file",
                               "white_rook_rank", "black_king_file", "black_king_rank"],
        "class_names": {"1": "draw", "10": "zero", "18": "one", "9": "two", "7": "three", "15": "four",
                        "14": "five", "3": "six", "2": "seven", "11": "eight", "17": "nine", "5": "ten",
                        "12": "eleven", "8": "twelve", "6": "thirteen", "16": "fourteen", "13": "fifteen", "4": "sixteen"},
        "relationship_basis": "relative positions, aligned files/ranks, distances and board boundaries",
    },
}
