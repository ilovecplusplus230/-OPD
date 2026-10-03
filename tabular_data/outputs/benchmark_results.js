// 自动生成：reference / reasoned / clean / seed42；最终分数为测试，逐轮轨迹为验证。
window.TabularBenchmarks = {
  "jungle_chess": {
    "name": "Jungle Chess RAW",
    "run_id": "5715996707fe45c2bb9cf4b01e4ec782",
    "baseline": {
      "f1": 0.5603241399240443,
      "log_loss": 0.5498473048210144
    },
    "final": {
      "f1": 0.7428160261986431,
      "log_loss": 0.45645204186439514
    },
    "improvement": {
      "f1": 18.25,
      "log_loss": -0.0934
    },
    "iter_f1": [
      0.5695832473792817,
      0.698141256623445,
      0.7070752199520519,
      0.7336903586705824,
      0.7379904352385758,
      0.7379904352385758
    ],
    "iter_log_loss": [
      0.5548086166381836,
      0.5182050466537476,
      0.4862666130065918,
      0.48220834136009216,
      0.46216925978660583,
      0.46216925978660583
    ],
    "rules": [
      {
        "desc": "black_piece0_strength 与 white_piece0_strength 的联合状态可能受到 white_piece0_file 的调制，第三列以饱和倒数改变交互强度。",
        "formula": "(df['black_piece0_strength'] - median_black_piece0_strength) / scale_black_piece0_strength * ((df['white_piece0_strength'] - median_white_piece0_strength) / scale_white_piece0_strength) / (1 + np.abs((df['white_piece0_file'] - median_white_piece0_file) / scale_white_piece0_file))"
      },
      {
        "desc": "检验简单交互能否解释当前模型尚未捕获的关系；与复杂候选一起预筛。",
        "formula": "df['white_piece0_strength'] - df['black_piece0_strength']"
      },
      {
        "desc": "重尾、饱和和反比例关系可能共同存在，需验证非线性复合是否改善分类。",
        "formula": "np.sign((df['oct_composite_19_r1'] - median_oct_composite_19_r1) / scale_oct_composite_19_r1) * np.log1p(np.abs((df['oct_composite_19_r1'] - median_oct_composite_19_r1) / scale_oct_composite_19_r1)) * np.tanh((df['white_piece0_strength'] - median_white_piece0_strength) / scale_white_piece0_strength) / (1 + np.abs((df['black_piece0_strength'] - median_black_piece0_strength) / scale_black_piece0_strength))"
      },
      {
        "desc": "双方棋子相对距离可能比各自绝对坐标更容易表达接触关系。",
        "formula": "np.abs(df['white_piece0_file'] - df['black_piece0_file']) + np.abs(df['white_piece0_rank'] - df['black_piece0_rank'])"
      }
    ],
    "top_features": [
      "black_piece0_strength",
      "white_piece0_strength",
      "white_piece0_file",
      "oct_composite_19_r1",
      "black_piece0_file",
      "white_piece0_rank",
      "black_piece0_rank"
    ]
  },
  "balance_scale": {
    "name": "Balance Scale",
    "run_id": "a3f6883ab6574f87b15e055bebb2b3d9",
    "baseline": {
      "f1": 0.5823596466229696,
      "log_loss": 0.507616400718689
    },
    "final": {
      "f1": 0.8047216587784539,
      "log_loss": 0.34705379605293274
    },
    "improvement": {
      "f1": 22.24,
      "log_loss": -0.1606
    },
    "iter_f1": [
      0.6438895045452422,
      0.7489566167460446,
      0.775219858950193,
      0.8063292277263328,
      0.8147692149846741,
      0.8147692149846741
    ],
    "iter_log_loss": [
      0.4336247444152832,
      0.39155200123786926,
      0.3606681525707245,
      0.3397432267665863,
      0.33662182092666626,
      0.33662182092666626
    ],
    "rules": [
      {
        "desc": "检验简单交互能否解释当前模型尚未捕获的关系；与复杂候选一起预筛。",
        "formula": "df['right-weight'] - df['left-weight']"
      },
      {
        "desc": "前两列的联合水平可能比乘性交互稳定，第三列调节其幅度。",
        "formula": "((df['oct_simple_difference_5_r1'] - median_oct_simple_difference_5_r1) / scale_oct_simple_difference_5_r1 + (df['right-distance'] - median_col_72696768742d64697374616e6365) / scale_col_72696768742d64697374616e6365) / (1 + np.abs((df['left-distance'] - median_col_6c6566742d64697374616e6365) / scale_col_6c6566742d64697374616e6365))"
      },
      {
        "desc": "重尾、饱和和反比例关系可能共同存在，需验证非线性复合是否改善分类。",
        "formula": "np.sign((df['oct_composite_sum_0_r2'] - median_oct_composite_sum_0_r2) / scale_oct_composite_sum_0_r2) * np.log1p(np.abs((df['oct_composite_sum_0_r2'] - median_oct_composite_sum_0_r2) / scale_oct_composite_sum_0_r2)) * np.tanh((df['right-distance'] - median_col_72696768742d64697374616e6365) / scale_col_72696768742d64697374616e6365) / (1 + np.abs((df['left-distance'] - median_col_6c6566742d64697374616e6365) / scale_col_6c6566742d64697374616e6365))"
      },
      {
        "desc": "前两列的联合水平可能比乘性交互稳定，第三列调节其幅度。",
        "formula": "((df['oct_composite_sum_0_r2'] - median_oct_composite_sum_0_r2) / scale_oct_composite_sum_0_r2 + (df['right-weight'] - median_col_72696768742d776569676874) / scale_col_72696768742d776569676874) / (1 + np.abs((df['left-weight'] - median_col_6c6566742d776569676874) / scale_col_6c6566742d776569676874))"
      }
    ],
    "top_features": [
      "right-weight",
      "left-weight",
      "oct_simple_difference_5_r1",
      "right-distance",
      "left-distance",
      "oct_composite_sum_0_r2"
    ]
  },
  "chess_krk": {
    "name": "Chess KRK",
    "run_id": "492b8b5c32564c81a18601a64f6afb16",
    "baseline": {
      "f1": 0.37048146954496675,
      "log_loss": 1.6418685913085938
    },
    "final": {
      "f1": 0.4351704783694546,
      "log_loss": 1.5811551809310913
    },
    "improvement": {
      "f1": 6.47,
      "log_loss": -0.0607
    },
    "iter_f1": [
      0.3547563882768442,
      0.40203175711163186,
      0.42895912154870336,
      0.4347673119816881,
      0.44090034022861574,
      0.44090034022861574
    ],
    "iter_log_loss": [
      1.6417877674102783,
      1.623831868171692,
      1.6051592826843262,
      1.5882987976074219,
      1.5762728452682495,
      1.5762728452682495
    ],
    "rules": [
      {
        "desc": "前两列的联合水平可能比乘性交互稳定，第三列调节其幅度。",
        "formula": "((df['white_king_rank'] - median_white_king_rank) / scale_white_king_rank + (df['black_king_file'] - median_black_king_file) / scale_black_king_file) / (1 + np.abs((df['white_rook_rank'] - median_white_rook_rank) / scale_white_rook_rank))"
      },
      {
        "desc": "前两列的联合水平可能比乘性交互稳定，第三列调节其幅度。",
        "formula": "((df['white_king_rank'] - median_white_king_rank) / scale_white_king_rank + (df['white_king_file'] - median_white_king_file) / scale_white_king_file) / (1 + np.abs((df['white_rook_file'] - median_white_rook_file) / scale_white_rook_file))"
      },
      {
        "desc": "检验简单交互能否解释当前模型尚未捕获的关系；与复杂候选一起预筛。",
        "formula": "(df['black_king_rank'] - median_black_king_rank) / scale_black_king_rank * ((df['white_rook_rank'] - median_white_rook_rank) / scale_white_rook_rank)"
      },
      {
        "desc": "第三列可能在不同条件区间表现为增强或抑制关系，分段特征将该假说交给验证集。",
        "formula": "np.where(df['oct_composite_sum_11_r1'] <= median_oct_composite_sum_11_r1, (df['black_king_file'] - median_black_king_file) / scale_black_king_file * ((df['white_rook_file'] - median_white_rook_file) / scale_white_rook_file), (df['black_king_file'] - median_black_king_file) / scale_black_king_file / (1 + np.abs((df['white_rook_file'] - median_white_rook_file) / scale_white_rook_file)))"
      }
    ],
    "top_features": [
      "white_king_rank",
      "black_king_file",
      "white_rook_rank",
      "white_king_file",
      "white_rook_file",
      "black_king_rank",
      "oct_composite_sum_11_r1"
    ]
  }
};
