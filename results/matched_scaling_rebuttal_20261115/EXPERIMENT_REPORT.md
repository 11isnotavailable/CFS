# CFS Matched-N / Fixed-K 补充实验报告

这是重新下载公开权重后的复现实验。Qwen 四规模使用新冻结的同一批 FineWeb-Edu contexts；不将其与历史 Qwen-14B 拼接。ViT 使用原模型系列、ImageNet-1K 验证集、原索引选择与预处理方法。未训练任何模型。

## 统计约定

Matched-N 是均匀无放回候选子集最大值的精确期望。Fixed-K 使用原 Cover（严格 CFS>0.5）及谱熵有效 Rank，除以 K；每输入/层抽样 500 次，seed 42，K=H 仅一次。原有有效 source 阈值与裁剪规则保持不变。

95% CI 为 2000 次输入级 percentile bootstrap。Oracle 在每层内按有效 source 等权、再按层等权；每次 bootstrap 重新计算分子/分母。相邻差异使用同一冻结输入的配对 bootstrap。Fixed-K 的子图 Monte Carlo SE 单独报告，不能把子图当作独立输入。所有 CI 均未作多重比较校正。

## 完整结果

### Qwen：4/4 个模型完成

| model | metric | mean | ci_low | ci_high | input_sd | subgraph_mc_se |
| --- | --- | --- | --- | --- | --- | --- |
| qwen14b | raw | 0.440904 | 0.411973 | 0.472802 | 0.087462 | 0.000000 |
| qwen14b | matched_2 | 0.271380 | 0.246692 | 0.298381 | 0.074604 | 0.000000 |
| qwen14b | matched_4 | 0.319086 | 0.292730 | 0.348040 | 0.079663 | 0.000000 |
| qwen14b | matched_8 | 0.361313 | 0.333828 | 0.392018 | 0.083331 | 0.000000 |
| qwen14b | matched_11 | 0.379088 | 0.351046 | 0.410387 | 0.084620 | 0.000000 |
| qwen14b | cover_full | 0.699479 | 0.632552 | 0.760423 | 0.186254 | 0.000000 |
| qwen14b | rank_full | 0.639290 | 0.605114 | 0.670851 | 0.095078 | 0.000000 |
| qwen14b | cover_4 | 0.881359 | 0.839566 | 0.917640 | 0.114646 | 0.000596 |
| qwen14b | rank_4 | 0.913510 | 0.896567 | 0.928001 | 0.045582 | 0.000203 |
| qwen14b | cover_8 | 0.827943 | 0.776648 | 0.874169 | 0.142612 | 0.000465 |
| qwen14b | rank_8 | 0.841619 | 0.815783 | 0.864560 | 0.071181 | 0.000198 |
| qwen14b | cover_12 | 0.795127 | 0.739045 | 0.845394 | 0.156047 | 0.000382 |
| qwen14b | rank_12 | 0.791969 | 0.762051 | 0.818999 | 0.082969 | 0.000174 |
| qwen15b | raw | 0.267861 | 0.253978 | 0.282113 | 0.042099 | 0.000000 |
| qwen15b | matched_2 | 0.160793 | 0.149812 | 0.172324 | 0.033871 | 0.000000 |
| qwen15b | matched_4 | 0.205488 | 0.193226 | 0.218498 | 0.037604 | 0.000000 |
| qwen15b | matched_8 | 0.248840 | 0.235480 | 0.262792 | 0.040632 | 0.000000 |
| qwen15b | matched_11 | 0.267861 | 0.253978 | 0.282113 | 0.042099 | 0.000000 |
| qwen15b | cover_full | 0.960069 | 0.943576 | 0.975694 | 0.046233 | 0.000000 |
| qwen15b | rank_full | 0.907174 | 0.896678 | 0.917519 | 0.031216 | 0.000000 |
| qwen15b | cover_4 | 0.986219 | 0.979620 | 0.991938 | 0.017880 | 0.000241 |
| qwen15b | rank_4 | 0.968483 | 0.964111 | 0.972591 | 0.012618 | 0.000091 |
| qwen15b | cover_8 | 0.972003 | 0.959677 | 0.982696 | 0.033592 | 0.000187 |
| qwen15b | rank_8 | 0.934868 | 0.926791 | 0.942649 | 0.023598 | 0.000076 |
| qwen15b | cover_12 | 0.960069 | 0.943576 | 0.975694 | 0.046233 | 0.000000 |
| qwen15b | rank_12 | 0.907174 | 0.896678 | 0.917519 | 0.031216 | 0.000000 |
| qwen3b | raw | 0.327258 | 0.308867 | 0.346459 | 0.055937 | 0.000000 |
| qwen3b | matched_2 | 0.203103 | 0.187849 | 0.218944 | 0.045834 | 0.000000 |
| qwen3b | matched_4 | 0.249177 | 0.232872 | 0.266096 | 0.049292 | 0.000000 |
| qwen3b | matched_8 | 0.291807 | 0.274286 | 0.309708 | 0.052661 | 0.000000 |
| qwen3b | matched_11 | 0.310175 | 0.292166 | 0.328736 | 0.054280 | 0.000000 |
| qwen3b | cover_full | 0.886719 | 0.853516 | 0.916683 | 0.093610 | 0.000000 |
| qwen3b | rank_full | 0.815789 | 0.798541 | 0.832590 | 0.051391 | 0.000000 |
| qwen3b | cover_4 | 0.954979 | 0.939731 | 0.968850 | 0.043123 | 0.000406 |
| qwen3b | rank_4 | 0.942277 | 0.935182 | 0.948939 | 0.020475 | 0.000135 |
| qwen3b | cover_8 | 0.923826 | 0.900612 | 0.944693 | 0.065576 | 0.000315 |
| qwen3b | rank_8 | 0.888111 | 0.876014 | 0.899678 | 0.035554 | 0.000125 |
| qwen3b | cover_12 | 0.902457 | 0.874236 | 0.928487 | 0.080859 | 0.000209 |
| qwen3b | rank_12 | 0.847637 | 0.832424 | 0.862393 | 0.045140 | 0.000088 |
| qwen7b | raw | 0.376968 | 0.356757 | 0.399670 | 0.063097 | 0.000000 |
| qwen7b | matched_2 | 0.225949 | 0.209176 | 0.244129 | 0.052291 | 0.000000 |
| qwen7b | matched_4 | 0.271810 | 0.253770 | 0.291733 | 0.056202 | 0.000000 |
| qwen7b | matched_8 | 0.313434 | 0.294459 | 0.334576 | 0.059242 | 0.000000 |
| qwen7b | matched_11 | 0.331246 | 0.311828 | 0.352706 | 0.060438 | 0.000000 |
| qwen7b | cover_full | 0.820312 | 0.771568 | 0.863477 | 0.138745 | 0.000000 |
| qwen7b | rank_full | 0.722353 | 0.698767 | 0.743789 | 0.066699 | 0.000000 |
| qwen7b | cover_4 | 0.935260 | 0.909202 | 0.955929 | 0.068977 | 0.000486 |
| qwen7b | rank_4 | 0.932959 | 0.923307 | 0.941229 | 0.026385 | 0.000162 |
| qwen7b | cover_8 | 0.900711 | 0.866387 | 0.928996 | 0.093243 | 0.000383 |
| qwen7b | rank_8 | 0.871714 | 0.856102 | 0.885413 | 0.043842 | 0.000165 |
| qwen7b | cover_12 | 0.877536 | 0.838849 | 0.910209 | 0.106640 | 0.000312 |
| qwen7b | rank_12 | 0.827641 | 0.808770 | 0.844569 | 0.053116 | 0.000146 |

### ViT：3/3 个模型完成

| model | metric | mean | ci_low | ci_high | input_sd | subgraph_mc_se |
| --- | --- | --- | --- | --- | --- | --- |
| vit_base | raw | 0.497663 | 0.485529 | 0.509679 | 0.132726 | 0.000000 |
| vit_base | matched_2 | 0.289034 | 0.277943 | 0.300356 | 0.122791 | 0.000000 |
| vit_base | matched_4 | 0.380751 | 0.368692 | 0.392803 | 0.132587 | 0.000000 |
| vit_base | matched_5 | 0.408722 | 0.396554 | 0.420869 | 0.133517 | 0.000000 |
| vit_base | cover_full | 0.598145 | 0.581543 | 0.614587 | 0.189012 | 0.000000 |
| vit_base | rank_full | 0.727230 | 0.715586 | 0.738670 | 0.130869 | 0.000000 |
| vit_base | cover_4 | 0.778778 | 0.766382 | 0.791084 | 0.140585 | 0.000327 |
| vit_base | rank_4 | 0.869644 | 0.861754 | 0.877243 | 0.089722 | 0.000176 |
| vit_base | cover_6 | 0.714204 | 0.699890 | 0.728446 | 0.161151 | 0.000263 |
| vit_base | rank_6 | 0.823683 | 0.814418 | 0.832863 | 0.105162 | 0.000138 |
| vit_large | raw | 0.603245 | 0.589460 | 0.617134 | 0.143638 | 0.000000 |
| vit_large | matched_2 | 0.373882 | 0.360237 | 0.387634 | 0.138818 | 0.000000 |
| vit_large | matched_4 | 0.468948 | 0.454612 | 0.483280 | 0.147099 | 0.000000 |
| vit_large | matched_5 | 0.496172 | 0.481826 | 0.510519 | 0.147601 | 0.000000 |
| vit_large | cover_full | 0.346802 | 0.329218 | 0.365726 | 0.217331 | 0.000000 |
| vit_large | rank_full | 0.465535 | 0.448938 | 0.481951 | 0.199126 | 0.000000 |
| vit_large | cover_4 | 0.541534 | 0.521669 | 0.560689 | 0.234579 | 0.000344 |
| vit_large | rank_4 | 0.637869 | 0.616927 | 0.657396 | 0.246764 | 0.000266 |
| vit_large | cover_6 | 0.478145 | 0.459321 | 0.497092 | 0.228084 | 0.000267 |
| vit_large | rank_6 | 0.591538 | 0.571577 | 0.610458 | 0.232811 | 0.000196 |
| vit_small | raw | 0.408621 | 0.395866 | 0.421992 | 0.153417 | 0.000000 |
| vit_small | matched_2 | 0.285045 | 0.273517 | 0.296977 | 0.134139 | 0.000000 |
| vit_small | matched_4 | 0.379312 | 0.366708 | 0.392189 | 0.149678 | 0.000000 |
| vit_small | matched_5 | 0.408621 | 0.395866 | 0.421992 | 0.153417 | 0.000000 |
| vit_small | cover_full | 0.729492 | 0.711255 | 0.747078 | 0.210127 | 0.000000 |
| vit_small | rank_full | 0.859644 | 0.850185 | 0.868622 | 0.105475 | 0.000000 |
| vit_small | cover_4 | 0.796081 | 0.781020 | 0.810245 | 0.172404 | 0.000255 |
| vit_small | rank_4 | 0.903886 | 0.896465 | 0.910635 | 0.081249 | 0.000096 |
| vit_small | cover_6 | 0.729492 | 0.711255 | 0.747078 | 0.210127 | 0.000000 |
| vit_small | rank_6 | 0.859644 | 0.850185 | 0.868622 | 0.105475 | 0.000000 |

## 相邻规模差异与单调性

| family | left | right | metric | delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen | qwen15b | qwen3b | cover_12 | -0.057613 | -0.081907 | -0.035848 |
| Qwen | qwen15b | qwen3b | cover_4 | -0.031240 | -0.044381 | -0.018614 |
| Qwen | qwen15b | qwen3b | cover_8 | -0.048177 | -0.068197 | -0.030114 |
| Qwen | qwen15b | qwen3b | cover_full | -0.073351 | -0.100917 | -0.047960 |
| Qwen | qwen15b | qwen3b | matched_11 | 0.042314 | 0.026603 | 0.058270 |
| Qwen | qwen15b | qwen3b | matched_2 | 0.042310 | 0.029154 | 0.055985 |
| Qwen | qwen15b | qwen3b | matched_4 | 0.043689 | 0.029980 | 0.058042 |
| Qwen | qwen15b | qwen3b | matched_8 | 0.042967 | 0.028055 | 0.058277 |
| Qwen | qwen15b | qwen3b | rank_12 | -0.059537 | -0.072637 | -0.047467 |
| Qwen | qwen15b | qwen3b | rank_4 | -0.026206 | -0.032141 | -0.020674 |
| Qwen | qwen15b | qwen3b | rank_8 | -0.046757 | -0.057085 | -0.037182 |
| Qwen | qwen15b | qwen3b | rank_full | -0.091386 | -0.105870 | -0.077826 |
| Qwen | qwen15b | qwen3b | raw | 0.059396 | 0.043073 | 0.075564 |
| Qwen | qwen3b | qwen7b | cover_12 | -0.024920 | -0.053737 | 0.002985 |
| Qwen | qwen3b | qwen7b | cover_4 | -0.019719 | -0.040696 | -0.001899 |
| Qwen | qwen3b | qwen7b | cover_8 | -0.023115 | -0.049648 | 0.001621 |
| Qwen | qwen3b | qwen7b | cover_full | -0.066406 | -0.102679 | -0.029746 |
| Qwen | qwen3b | qwen7b | matched_11 | 0.021071 | 0.005009 | 0.037748 |
| Qwen | qwen3b | qwen7b | matched_2 | 0.022846 | 0.006942 | 0.039250 |
| Qwen | qwen3b | qwen7b | matched_4 | 0.022633 | 0.006711 | 0.038945 |
| Qwen | qwen3b | qwen7b | matched_8 | 0.021627 | 0.005445 | 0.037988 |
| Qwen | qwen3b | qwen7b | rank_12 | -0.019996 | -0.035777 | -0.004280 |
| Qwen | qwen3b | qwen7b | rank_4 | -0.009318 | -0.017138 | -0.001771 |
| Qwen | qwen3b | qwen7b | rank_8 | -0.016397 | -0.029495 | -0.003576 |
| Qwen | qwen3b | qwen7b | rank_full | -0.093436 | -0.112481 | -0.074815 |
| Qwen | qwen3b | qwen7b | raw | 0.049710 | 0.033473 | 0.066429 |
| Qwen | qwen7b | qwen14b | cover_12 | -0.082410 | -0.131323 | -0.039615 |
| Qwen | qwen7b | qwen14b | cover_4 | -0.053901 | -0.090272 | -0.022204 |
| Qwen | qwen7b | qwen14b | cover_8 | -0.072768 | -0.118112 | -0.033366 |
| Qwen | qwen7b | qwen14b | cover_full | -0.120833 | -0.175299 | -0.071566 |
| Qwen | qwen7b | qwen14b | matched_11 | 0.047842 | 0.024871 | 0.074068 |
| Qwen | qwen7b | qwen14b | matched_2 | 0.045430 | 0.023726 | 0.070627 |
| Qwen | qwen7b | qwen14b | matched_4 | 0.047276 | 0.024757 | 0.072707 |
| Qwen | qwen7b | qwen14b | matched_8 | 0.047879 | 0.025299 | 0.073967 |
| Qwen | qwen7b | qwen14b | rank_12 | -0.035671 | -0.063028 | -0.012178 |
| Qwen | qwen7b | qwen14b | rank_4 | -0.019449 | -0.035088 | -0.006693 |
| Qwen | qwen7b | qwen14b | rank_8 | -0.030095 | -0.053871 | -0.009772 |
| Qwen | qwen7b | qwen14b | rank_full | -0.083063 | -0.113700 | -0.055676 |
| Qwen | qwen7b | qwen14b | raw | 0.063936 | 0.041262 | 0.089950 |
| ViT | vit_small | vit_base | cover_4 | -0.017303 | -0.031708 | -0.002549 |
| ViT | vit_small | vit_base | cover_6 | -0.015288 | -0.032784 | 0.002648 |
| ViT | vit_small | vit_base | cover_full | -0.131348 | -0.150553 | -0.111650 |
| ViT | vit_small | vit_base | matched_2 | 0.003989 | -0.007137 | 0.014665 |
| ViT | vit_small | vit_base | matched_4 | 0.001438 | -0.010580 | 0.013274 |
| ViT | vit_small | vit_base | matched_5 | 0.000101 | -0.012018 | 0.012171 |
| ViT | vit_small | vit_base | rank_4 | -0.034242 | -0.042470 | -0.025941 |
| ViT | vit_small | vit_base | rank_6 | -0.035961 | -0.046021 | -0.025845 |
| ViT | vit_small | vit_base | rank_full | -0.132414 | -0.143539 | -0.121247 |
| ViT | vit_small | vit_base | raw | 0.089042 | 0.076655 | 0.101460 |
| ViT | vit_base | vit_large | cover_4 | -0.237244 | -0.256246 | -0.218090 |
| ViT | vit_base | vit_large | cover_6 | -0.236059 | -0.255426 | -0.216996 |
| ViT | vit_base | vit_large | cover_full | -0.251343 | -0.270550 | -0.232297 |
| ViT | vit_base | vit_large | matched_2 | 0.084848 | 0.072687 | 0.096599 |
| ViT | vit_base | vit_large | matched_4 | 0.088197 | 0.075482 | 0.100589 |
| ViT | vit_base | vit_large | matched_5 | 0.087450 | 0.074753 | 0.099817 |
| ViT | vit_base | vit_large | rank_4 | -0.231775 | -0.251282 | -0.212519 |
| ViT | vit_base | vit_large | rank_6 | -0.232145 | -0.250837 | -0.213457 |
| ViT | vit_base | vit_large | rank_full | -0.261695 | -0.279410 | -0.244317 |
| ViT | vit_base | vit_large | raw | 0.105582 | 0.093411 | 0.117978 |

| family | metric | models | monotonic_increasing | monotonic_decreasing |
| --- | --- | --- | --- | --- |
| Qwen | cover_12 | 4 | False | True |
| Qwen | cover_4 | 4 | False | True |
| Qwen | cover_8 | 4 | False | True |
| Qwen | cover_full | 4 | False | True |
| Qwen | matched_11 | 4 | True | False |
| Qwen | matched_2 | 4 | True | False |
| Qwen | matched_4 | 4 | True | False |
| Qwen | matched_8 | 4 | True | False |
| Qwen | rank_12 | 4 | False | True |
| Qwen | rank_4 | 4 | False | True |
| Qwen | rank_8 | 4 | False | True |
| Qwen | rank_full | 4 | False | True |
| Qwen | raw | 4 | True | False |
| ViT | cover_4 | 3 | False | True |
| ViT | cover_6 | 3 | False | True |
| ViT | cover_full | 3 | False | True |
| ViT | matched_2 | 3 | True | False |
| ViT | matched_4 | 3 | True | False |
| ViT | matched_5 | 3 | True | False |
| ViT | rank_4 | 3 | False | True |
| ViT | rank_6 | 3 | False | True |
| ViT | rank_full | 3 | False | True |
| ViT | raw | 3 | True | False |

## 与原实现及原论文表的一致性

| model | valid_source_fraction | matched_2_max_absolute_error | K_equals_H_checks |
| --- | --- | --- | --- |
| qwen14b | 1.000000 | 0.000000 | not applicable |
| qwen15b | 1.000000 | 0.000000 | cover,rank |
| qwen3b | 1.000000 | 0.000000 | not applicable |
| qwen7b | 1.000000 | 0.000000 | not applicable |
| vit_base | 0.967448 | 0.000000 | not applicable |
| vit_large | 0.744385 | 0.000000 | not applicable |
| vit_small | 0.996419 | 0.000000 | cover,rank |

| family | model | metric | historical | rerun | delta |
| --- | --- | --- | --- | --- | --- |
| Qwen | qwen14b | raw | 0.443000 | 0.440904 | -0.002096 |
| Qwen | qwen14b | matched_2 | 0.274500 | 0.271380 | -0.003120 |
| Qwen | qwen14b | cover_full | 0.699200 | 0.699479 | 0.000279 |
| Qwen | qwen14b | rank_full | 0.641893 | 0.639290 | -0.002603 |
| Qwen | qwen15b | raw | 0.272519 | 0.267861 | -0.004658 |
| Qwen | qwen15b | matched_2 | 0.162005 | 0.160793 | -0.001213 |
| Qwen | qwen15b | cover_full | 0.954861 | 0.960069 | 0.005208 |
| Qwen | qwen15b | rank_full | 0.905212 | 0.907174 | 0.001962 |
| Qwen | qwen3b | raw | 0.319844 | 0.327258 | 0.007413 |
| Qwen | qwen3b | matched_2 | 0.195261 | 0.203103 | 0.007842 |
| Qwen | qwen3b | cover_full | 0.898438 | 0.886719 | -0.011719 |
| Qwen | qwen3b | rank_full | 0.826594 | 0.815789 | -0.010805 |
| Qwen | qwen7b | raw | 0.371234 | 0.376968 | 0.005734 |
| Qwen | qwen7b | matched_2 | 0.218042 | 0.225949 | 0.007907 |
| Qwen | qwen7b | cover_full | 0.824033 | 0.820312 | -0.003720 |
| Qwen | qwen7b | rank_full | 0.742685 | 0.722353 | -0.020332 |
| ViT | vit_base | raw | 0.497607 | 0.497663 | 0.000056 |
| ViT | vit_base | matched_2 | 0.288869 | 0.289034 | 0.000165 |
| ViT | vit_base | cover_full | 0.597168 | 0.598145 | 0.000977 |
| ViT | vit_base | rank_full | 0.727559 | 0.727230 | -0.000329 |
| ViT | vit_large | raw | 0.603441 | 0.603245 | -0.000196 |
| ViT | vit_large | matched_2 | 0.373709 | 0.373882 | 0.000173 |
| ViT | vit_large | cover_full | 0.345459 | 0.346802 | 0.001343 |
| ViT | vit_large | rank_full | 0.464672 | 0.465535 | 0.000863 |
| ViT | vit_small | raw | 0.408754 | 0.408621 | -0.000133 |
| ViT | vit_small | matched_2 | 0.285242 | 0.285045 | -0.000198 |
| ViT | vit_small | cover_full | 0.732096 | 0.729492 | -0.002604 |
| ViT | vit_small | rank_full | 0.860192 | 0.859644 | -0.000548 |

同矩阵对照检查实现一致性；与论文汇总表的差异还包含输入、硬件和数值环境差异。历史 Qwen 输入缺失，不能对历史表做逐输入配对，也不能把接近的数值写成完全复现。

## 抽样稳定性

| model | metric | draws | mean |
| --- | --- | --- | --- |
| qwen14b | cover_12 | 100 | 0.795356 |
| qwen14b | cover_12 | 250 | 0.795535 |
| qwen14b | cover_12 | 500 | 0.795127 |
| qwen14b | cover_4 | 100 | 0.883047 |
| qwen14b | cover_4 | 250 | 0.881094 |
| qwen14b | cover_4 | 500 | 0.881359 |
| qwen14b | cover_8 | 100 | 0.826576 |
| qwen14b | cover_8 | 250 | 0.827531 |
| qwen14b | cover_8 | 500 | 0.827943 |
| qwen14b | rank_12 | 100 | 0.792007 |
| qwen14b | rank_12 | 250 | 0.792299 |
| qwen14b | rank_12 | 500 | 0.791969 |
| qwen14b | rank_4 | 100 | 0.914158 |
| qwen14b | rank_4 | 250 | 0.913672 |
| qwen14b | rank_4 | 500 | 0.913510 |
| qwen14b | rank_8 | 100 | 0.841156 |
| qwen14b | rank_8 | 250 | 0.841392 |
| qwen14b | rank_8 | 500 | 0.841619 |
| qwen15b | cover_12 | 1 | 0.960069 |
| qwen15b | cover_4 | 100 | 0.986458 |
| qwen15b | cover_4 | 250 | 0.986052 |
| qwen15b | cover_4 | 500 | 0.986219 |
| qwen15b | cover_8 | 100 | 0.971719 |
| qwen15b | cover_8 | 250 | 0.971833 |
| qwen15b | cover_8 | 500 | 0.972003 |
| qwen15b | rank_12 | 1 | 0.907174 |
| qwen15b | rank_4 | 100 | 0.968165 |
| qwen15b | rank_4 | 250 | 0.968354 |
| qwen15b | rank_4 | 500 | 0.968483 |
| qwen15b | rank_8 | 100 | 0.935029 |
| qwen15b | rank_8 | 250 | 0.934974 |
| qwen15b | rank_8 | 500 | 0.934868 |
| qwen3b | cover_12 | 100 | 0.902214 |
| qwen3b | cover_12 | 250 | 0.902198 |
| qwen3b | cover_12 | 500 | 0.902457 |
| qwen3b | cover_4 | 100 | 0.954453 |
| qwen3b | cover_4 | 250 | 0.955063 |
| qwen3b | cover_4 | 500 | 0.954979 |
| qwen3b | cover_8 | 100 | 0.923568 |
| qwen3b | cover_8 | 250 | 0.923750 |
| qwen3b | cover_8 | 500 | 0.923826 |
| qwen3b | rank_12 | 100 | 0.847503 |
| qwen3b | rank_12 | 250 | 0.847638 |
| qwen3b | rank_12 | 500 | 0.847637 |
| qwen3b | rank_4 | 100 | 0.942212 |
| qwen3b | rank_4 | 250 | 0.942270 |
| qwen3b | rank_4 | 500 | 0.942277 |
| qwen3b | rank_8 | 100 | 0.887862 |
| qwen3b | rank_8 | 250 | 0.888101 |
| qwen3b | rank_8 | 500 | 0.888111 |
| qwen7b | cover_12 | 100 | 0.876450 |
| qwen7b | cover_12 | 250 | 0.877080 |
| qwen7b | cover_12 | 500 | 0.877536 |
| qwen7b | cover_4 | 100 | 0.934818 |
| qwen7b | cover_4 | 250 | 0.935510 |
| qwen7b | cover_4 | 500 | 0.935260 |
| qwen7b | cover_8 | 100 | 0.900859 |
| qwen7b | cover_8 | 250 | 0.900797 |
| qwen7b | cover_8 | 500 | 0.900711 |
| qwen7b | rank_12 | 100 | 0.827255 |
| qwen7b | rank_12 | 250 | 0.827418 |
| qwen7b | rank_12 | 500 | 0.827641 |
| qwen7b | rank_4 | 100 | 0.932882 |
| qwen7b | rank_4 | 250 | 0.932969 |
| qwen7b | rank_4 | 500 | 0.932959 |
| qwen7b | rank_8 | 100 | 0.871984 |
| qwen7b | rank_8 | 250 | 0.871891 |
| qwen7b | rank_8 | 500 | 0.871714 |
| vit_base | cover_4 | 100 | 0.778091 |
| vit_base | cover_4 | 250 | 0.778189 |
| vit_base | cover_4 | 500 | 0.778778 |
| vit_base | cover_6 | 100 | 0.714082 |
| vit_base | cover_6 | 250 | 0.714467 |
| vit_base | cover_6 | 500 | 0.714204 |
| vit_base | rank_4 | 100 | 0.869472 |
| vit_base | rank_4 | 250 | 0.869232 |
| vit_base | rank_4 | 500 | 0.869644 |
| vit_base | rank_6 | 100 | 0.823640 |
| vit_base | rank_6 | 250 | 0.823606 |
| vit_base | rank_6 | 500 | 0.823683 |
| vit_large | cover_4 | 100 | 0.541885 |
| vit_large | cover_4 | 250 | 0.541441 |
| vit_large | cover_4 | 500 | 0.541534 |
| vit_large | cover_6 | 100 | 0.477744 |
| vit_large | cover_6 | 250 | 0.478031 |
| vit_large | cover_6 | 500 | 0.478145 |
| vit_large | rank_4 | 100 | 0.638197 |
| vit_large | rank_4 | 250 | 0.637853 |
| vit_large | rank_4 | 500 | 0.637869 |
| vit_large | rank_6 | 100 | 0.591673 |
| vit_large | rank_6 | 250 | 0.591685 |
| vit_large | rank_6 | 500 | 0.591538 |
| vit_small | cover_4 | 100 | 0.795439 |
| vit_small | cover_4 | 250 | 0.795861 |
| vit_small | cover_4 | 500 | 0.796081 |
| vit_small | cover_6 | 1 | 0.729492 |
| vit_small | rank_4 | 100 | 0.903751 |
| vit_small | rank_4 | 250 | 0.903859 |
| vit_small | rank_4 | 500 | 0.903886 |
| vit_small | rank_6 | 1 | 0.859644 |

## Q1：固定候选数后的趋势

Qwen matched_11 的均值随规模单调增加。
Qwen matched_2 的均值随规模单调增加。
Qwen matched_4 的均值随规模单调增加。
Qwen matched_8 的均值随规模单调增加。

ViT matched_2 的均值随规模单调增加。
ViT matched_4 的均值随规模单调增加。
ViT matched_5 的均值随规模单调增加。

## Q2：固定子图规模后的趋势

Qwen cover_12 的均值随规模单调减少。
Qwen cover_4 的均值随规模单调减少。
Qwen cover_8 的均值随规模单调减少。
Qwen rank_12 的均值随规模单调减少。
Qwen rank_4 的均值随规模单调减少。
Qwen rank_8 的均值随规模单调减少。
ViT cover_4 的均值随规模单调减少。
ViT cover_6 的均值随规模单调减少。
ViT rank_4 的均值随规模单调减少。
ViT rank_6 的均值随规模单调减少。

## Q3 与 Reviewer nLkz 的回应依据

Qwen（完整系列）：
固定候选数共有 12 个相邻规模/配置对照，其中 0 个配对 95% CI 包含零。
固定子图 Cover共有 9 个相邻规模/配置对照，其中 2 个配对 95% CI 包含零。
这些对照尚不足以支持明确的相邻规模差异：qwen3b→qwen7b cover_12；qwen3b→qwen7b cover_8。
固定子图 Rank共有 9 个相邻规模/配置对照，其中 0 个配对 95% CI 包含零。

ViT（完整系列）：
固定候选数共有 6 个相邻规模/配置对照，其中 3 个配对 95% CI 包含零。
这些对照尚不足以支持明确的相邻规模差异：vit_small→vit_base matched_2；vit_small→vit_base matched_4；vit_small→vit_base matched_5。
固定子图 Cover共有 4 个相邻规模/配置对照，其中 1 个配对 95% CI 包含零。
这些对照尚不足以支持明确的相邻规模差异：vit_small→vit_base cover_6。
固定子图 Rank共有 4 个相邻规模/配置对照，其中 0 个配对 95% CI 包含零。

应逐个 N/K 对照上表和配对差异 CI：若控制后保持原方向，可称该已测趋势对相应候选数或子图规模控制稳健；若方向改变、非单调或相邻差异 CI 包含零，应如实弱化相应结论。不能仅凭均值排序或少数规模点的相关系数声称统计显著。

这些控制排除了当前设置下的一部分候选数量解释，但没有固定模型深度、宽度、GQA、预训练或有效 source 比例。特别是 ViT-L 的 validity 过滤可能影响 Cover/K 和 Rank/K。结果仍是模型系列内的描述性关联，不是规模导致功能替代性的因果证据，也不是可执行联合压缩率或任务效用证明。

图表：`figure1_matched_N.pdf/png`、`figure2_fixed_K.pdf/png`；完整数值 `all_results.csv/json/tex`，配对差异 `paired_contrasts.csv`，逐输入与全部 bootstrap replicate 位于各模型分析目录。
