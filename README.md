# CDGON
Code and Data for the paper "Physics-informed NeuralODE for Post-disaster Mobility Recovery", accepted by KDD2024 Research Track.



# Supplementary materials, including proof of convergence of the key formula $\frac{\mathrm{d} r_i(t)}{\mathrm{d} t}  = \alpha \frac{r_i(t)}{\overline{r_i}}[ \overline{r_i} - r_i(t)]$, performance evaluation results of STGCN and CDGON, and hyper-parameter experimental results:



## Theorem: In the formula $\frac{\mathrm{d} r_i(t)}{\mathrm{d} t}  = \alpha \frac{r_i(t)}{\overline{r_i}}[ \overline{r_i} - r_i(t)]$, $r_i(t)$ converges to $\overline{r_i}$ when $t \to \infty $ instead of oscillating perpetually around $\overline{r_i}$.

**Proof of Theorem:**

The differential equation $\frac{\mathrm{d} r_i(t)}{\mathrm{d} t}  = \alpha \frac{r_i(t)}{\overline{r_i}}[ \overline{r_i} - r_i(t)]$ can be transformed to:

$\overline{r_i}\mathrm{d} r_i(t)  = \alpha r_i(t)[ \overline{r_i} - r_i(t)] \mathrm{d} t$,

which is a separable variable equation and we can have:

$\frac{1}{\alpha r_i(t)[ \overline{r_i} - r_i(t)]}\mathrm{d} r_i(t)  =  \frac{1}{\overline{r_i}}\mathrm{d} t$,

which can be integrated to obtain:

$\frac{r_i(t)}{\overline{r_i} - r_i(t)} = Ce^{\alpha t}$,

which can generate:

$r_i(t) = \frac{\overline{r_i}}{1+Ce^{-\alpha t}} $,

which satisfies:

$\lim_{t \to \infty} r_i(t) = \lim_{t \to \infty}\frac{\overline{r_i}}{1+Ce^{-\alpha t}}=\overline{r_i}$

which proves the theorem





## Comparision of Performance Evaluation between STGCN and CDGON



<table>
    <tr>
        <td>Experiments Type</td>
        <td>Metrics</td>
        <td>STGCN</td>
        <td>CDGON</td>
    </tr>
    <tr>
        <td rowspan=3>Performance Evaluation in FL</td>
        <td>MAE</td>
        <td>62417.8359</td>
        <td>59767.4805</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9909</td>
        <td>0.9948</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.0954</td>
        <td>0.0724</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Performance Evaluation in GA</td>
        <td>MAE</td>
        <td>7082.8677</td>
        <td>2013.2821</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9734</td>
        <td>0.9977</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1631</td>
        <td>0.0475</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Performance Evaluation in in SC</td>
        <td>MAE</td>
        <td>18133.7500</td>
        <td>9040.6785</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9759</td>
        <td>0.9941</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1554</td>
        <td>0.0771</td>
    </tr>
    <tr>
<table>


## Hyper-parameter experimental results

**Hyper-parameter settings in CDGON**

*Embedding dimension*: 48

*edge loss weight $\lambda$*: 100

*Learning rate*: 0.003

### Experiment results on different embedding dimensions, where the other parameter settings are same as original paper.
<table>
    <tr>
        <td>Experiments Type</td>
        <td>Metrics</td>
        <td>16</td>
        <td>32</td>
        <td>48</td>
        <td>64</td>
        <td>80</td>
    </tr>
    <tr>
        <td rowspan=3>Performance Evaluation in FL</td>
        <td>MAE</td>
        <td>65796.6484</td>
        <td>31556.1035</td>
        <td>59767.4805</td>
        <td>38900.8594</td>
        <td>16794.2266</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9817</td>
        <td>0.9947</td>
        <td>0.9948</td>
        <td>0.9949</td>
        <td>0.9993</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1353</td>
        <td>0.0728</td>
        <td>0.0724</td>
        <td>0.0711</td>
        <td>0.0265</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Performance Evaluation in GA</td>
        <td>MAE</td>
        <td>8596.8066</td>
        <td>2476.3357</td>
        <td>2013.2821</td>
        <td>6726.749</td>
        <td>2004.8026</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9</td>
        <td>0.9965</td>
        <td>0.9977</td>
        <td>0.9649</td>
        <td>0.9976</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.3162</td>
        <td>0.0588</td>
        <td>0.0475</td>
        <td>0.1873</td>
        <td>0.0493</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Performance Evaluation in in SC</td>
        <td>MAE</td>
        <td>15904.1963</td>
        <td>20676.6699</td>
        <td>9040.6758</td>
        <td>13369.793</td>
        <td>35273.7461</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9814</td>
        <td>0.9683</td>
        <td>0.9941</td>
        <td>0.9821</td>
        <td>0.9332</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1363</td>
        <td>0.178</td>
        <td>0.0771</td>
        <td>0.134</td>
        <td>0.2584</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization FL -> GA</td>
        <td>MAE</td>
        <td>8991.2666</td>
        <td>5983.022</td>
        <td>5433.7192</td>
        <td>3936.6899</td>
        <td>12559.5605</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9442</td>
        <td>0.9704</td>
        <td>0.9831</td>
        <td>0.9852</td>
        <td>0.8773</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.2363</td>
        <td>0.1722</td>
        <td>0.1301</td>
        <td>0.1215</td>
        <td>0.3502</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization FL -> SC</td>
        <td>MAE</td>
        <td>22449.4219</td>
        <td>15979.0742</td>
        <td>13609.5889</td>
        <td>13573.167</td>
        <td>32977.6367</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9525</td>
        <td>0.9736</td>
        <td>0.9773</td>
        <td>0.9797</td>
        <td>0.8973</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.218</td>
        <td>0.1625</td>
        <td>0.1508</td>
        <td>0.1426</td>
        <td>0.3204</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization GA -> FL</td>
        <td>MAE</td>
        <td>44399.207</td>
        <td>46987.5703</td>
        <td>48276.1992</td>
        <td>40045.4766</td>
        <td>57892.6836</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9901</td>
        <td>0.9919</td>
        <td>0.9901</td>
        <td>0.9922</td>
        <td>0.9863</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.0997</td>
        <td>0.0898</td>
        <td>0.0997</td>
        <td>0.0881</td>
        <td>0.1171</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization GA -> SC</td>
        <td>MAE</td>
        <td>14188.7959</td>
        <td>16384.1875</td>
        <td>14315.9375</td>
        <td>12567.7197</td>
        <td>13666.5508</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9721</td>
        <td>0.9732</td>
        <td>0.982</td>
        <td>0.9847</td>
        <td>0.9831</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.167</td>
        <td>0.1637</td>
        <td>0.1341</td>
        <td>0.1235</td>
        <td>0.13</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization SC -> FL</td>
        <td>MAE</td>
        <td>37216.1172</td>
        <td>42591.7539</td>
        <td>73204.6719</td>
        <td>53571.0898</td>
        <td>74730.8906</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9926</td>
        <td>0.9916</td>
        <td>0.9801</td>
        <td>0.988</td>
        <td>0.9782</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.0859</td>
        <td>0.0916</td>
        <td>0.1412</td>
        <td>0.1095</td>
        <td>0.1478</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization SC -> GA</td>
        <td>MAE</td>
        <td>4730.644</td>
        <td>3934.5408</td>
        <td>8921.1035</td>
        <td>3830.97</td>
        <td>11805.2217</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9785</td>
        <td>0.9875</td>
        <td>0.9687</td>
        <td>0.9891</td>
        <td>0.9456</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1467</td>
        <td>0.1117</td>
        <td>0.177</td>
        <td>0.1042</td>
        <td>0.2331</td>
    </tr>
    <tr>
</table>

### Experiment results on different $\lambda$, where the other parameter settings are same as original paper.
<table>
    <tr>
        <td>Experiments</td>
        <td>Metrics</td>
        <td>10</td>
        <td>50</td>
        <td>100</td>
        <td>500</td>
        <td>1000</td>
    </tr>
    <tr>
        <td rowspan=3>Performance Evaluation in FL</td>
        <td>MAE</td>
        <td>32522.3301</td>
        <td>40861.3164</td>
        <td>59767.4805</td>
        <td>35012.9688</td>
        <td>64905.2461</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.994</td>
        <td>0.9912</td>
        <td>0.9948</td>
        <td>0.9953</td>
        <td>0.9866</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.0772</td>
        <td>0.0939</td>
        <td>0.0724</td>
        <td>0.0686</td>
        <td>0.116</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Performance Evaluation in GA</td>
        <td>MAE</td>
        <td>43049.1211</td>
        <td>4385.7549</td>
        <td>2013.2821</td>
        <td>7148.8594</td>
        <td>5631.6104</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.4618</td>
        <td>0.9816</td>
        <td>0.9977</td>
        <td>0.9399</td>
        <td>0.9876</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.7336</td>
        <td>0.1357</td>
        <td>0.0475</td>
        <td>0.2452</td>
        <td>0.1113</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Performance Evaluation in SC</td>
        <td>MAE</td>
        <td>21663.9062</td>
        <td>22532.2617</td>
        <td>9040.6758</td>
        <td>16925.8652</td>
        <td>38515.3086</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9648</td>
        <td>0.9658</td>
        <td>0.9941</td>
        <td>0.9674</td>
        <td>0.8788</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1875</td>
        <td>0.1848</td>
        <td>0.0771</td>
        <td>0.1804</td>
        <td>0.3481</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization FL -> GA</td>
        <td>MAE</td>
        <td>7073.2612</td>
        <td>7787.6982</td>
        <td>5433.7192</td>
        <td>4014.4309</td>
        <td>8322.3545</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9646</td>
        <td>0.9603</td>
        <td>0.9831</td>
        <td>0.9835</td>
        <td>0.9553</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1882</td>
        <td>0.1993</td>
        <td>0.1301</td>
        <td>0.1283</td>
        <td>0.2114</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization FL -> SC</td>
        <td>MAE</td>
        <td>17733.6484</td>
        <td>18547.4883</td>
        <td>13609.5889</td>
        <td>14024.291</td>
        <td>21716.9141</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9692</td>
        <td>0.9665</td>
        <td>0.9773</td>
        <td>0.9771</td>
        <td>0.9559</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1756</td>
        <td>0.1832</td>
        <td>0.1508</td>
        <td>0.1513</td>
        <td>0.2099</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization GA -> FL</td>
        <td>MAE</td>
        <td>44205.918</td>
        <td>45013.5977</td>
        <td>48276.1992</td>
        <td>50217.8906</td>
        <td>52088.3398</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9913</td>
        <td>0.9912</td>
        <td>0.9901</td>
        <td>0.9889</td>
        <td>0.9904</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.0932</td>
        <td>0.0937</td>
        <td>0.0997</td>
        <td>0.1053</td>
        <td>0.0979</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization GA -> SC</td>
        <td>MAE</td>
        <td>12889.4541</td>
        <td>13916.4736</td>
        <td>14315.9375</td>
        <td>13024.3271</td>
        <td>12746.8926</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9849</td>
        <td>0.9823</td>
        <td>0.982</td>
        <td>0.986</td>
        <td>0.986</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1227</td>
        <td>0.1329</td>
        <td>0.1341</td>
        <td>0.1185</td>
        <td>0.1184</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization SC -> FL</td>
        <td>MAE</td>
        <td>67278.0703</td>
        <td>80161.0156</td>
        <td>73204.6719</td>
        <td>54324.0898</td>
        <td>69783.5703</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9834</td>
        <td>0.9756</td>
        <td>0.9801</td>
        <td>0.9888</td>
        <td>0.9717</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1288</td>
        <td>0.1561</td>
        <td>0.1412</td>
        <td>0.1058</td>
        <td>0.1681</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization SC -> GA</td>
        <td>MAE</td>
        <td>5476.9692</td>
        <td>14294.9346</td>
        <td>8921.1035</td>
        <td>3708.6938</td>
        <td>9658.623</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.9864</td>
        <td>0.9228</td>
        <td>0.9687</td>
        <td>0.9922</td>
        <td>0.9241</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.1166</td>
        <td>0.2779</td>
        <td>0.177</td>
        <td>0.0881</td>
        <td>0.2756</td>
    </tr>
    <tr>
</table>

### Experiment results on different learning rates, where the other parameter settings are same as original paper.

<table>
    <tr>
        <td>Experiments</td>
        <td>Metrics</td>
        <td>0.0001</td>
        <td>0.0005</td>
        <td>0.001</td>
        <td>0.005</td>
        <td>0.01</td>
    </tr>
    <tr>
        <td rowspan=3>Performance Evaluation in FL</td>
        <td>MAE</td>
        <td>529004.1875</td>
        <td>34693.8633</td>
        <td>33831.0703</td>
        <td>36524.457</td>
        <td>60011.1562</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.1695</td>
        <td>0.9975</td>
        <td>0.9976</td>
        <td>0.9984</td>
        <td>0.9942</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.9113</td>
        <td>0.0496</td>
        <td>0.0488</td>
        <td>0.0396</td>
        <td>0.076</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Performance Evaluation in GA</td>
        <td>MAE</td>
        <td>37850.4648</td>
        <td>4462.7021</td>
        <td>5688.2329</td>
        <td>2161.2673</td>
        <td>11902.7266</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.003</td>
        <td>0.9876</td>
        <td>0.9777</td>
        <td>0.9972</td>
        <td>0.9387</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.9985</td>
        <td>0.1112</td>
        <td>0.1495</td>
        <td>0.0526</td>
        <td>0.2476</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Performance Evaluation in SC</td>
        <td>MAE</td>
        <td>58461.9492</td>
        <td>15060.959</td>
        <td>18775.9277</td>
        <td>32302.1719</td>
        <td>19255.9629</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.7281</td>
        <td>0.9858</td>
        <td>0.9748</td>
        <td>0.9269</td>
        <td>0.9672</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.5214</td>
        <td>0.1194</td>
        <td>0.1586</td>
        <td>0.2703</td>
        <td>0.1812</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization FL -> GA</td>
        <td>MAE</td>
        <td>46940.0625</td>
        <td>39384.4102</td>
        <td>6600.5337</td>
        <td>4510.082</td>
        <td>5471.5781</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>-0.3108</td>
        <td>0.0531</td>
        <td>0.9146</td>
        <td>0.9828</td>
        <td>0.9798</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>1.1449</td>
        <td>0.9731</td>
        <td>0.2923</td>
        <td>0.1311</td>
        <td>0.1421</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization FL -> SC</td>
        <td>MAE</td>
        <td>151936.8438</td>
        <td>131516.125</td>
        <td>19109.916</td>
        <td>17394.168</td>
        <td>15692.9873</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>-0.6068</td>
        <td>-0.2168</td>
        <td>0.9596</td>
        <td>0.9638</td>
        <td>0.9707</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>1.2676</td>
        <td>1.1031</td>
        <td>0.201</td>
        <td>0.1904</td>
        <td>0.1712</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization GA -> FL</td>
        <td>MAE</td>
        <td>437491.4062</td>
        <td>44095.7656</td>
        <td>44387.4062</td>
        <td>53430.0625</td>
        <td>68644.3203</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.2801</td>
        <td>0.9919</td>
        <td>0.9919</td>
        <td>0.9883</td>
        <td>0.9839</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.8485</td>
        <td>0.0901</td>
        <td>0.0897</td>
        <td>0.1083</td>
        <td>0.1271</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization GA -> SC</td>
        <td>MAE</td>
        <td>110432.5</td>
        <td>14388.5625</td>
        <td>13461.3564</td>
        <td>13739.5166</td>
        <td>14401.3418</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.1196</td>
        <td>0.9806</td>
        <td>0.9821</td>
        <td>0.9843</td>
        <td>0.9798</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.9383</td>
        <td>0.1392</td>
        <td>0.1339</td>
        <td>0.1254</td>
        <td>0.142</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization SC -> FL</td>
        <td>MAE</td>
        <td>424390.75</td>
        <td>47684.0977</td>
        <td>90427.2734</td>
        <td>84410.2734</td>
        <td>83852.9453</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.3208</td>
        <td>0.9903</td>
        <td>0.9352</td>
        <td>0.968</td>
        <td>0.9719</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.8241</td>
        <td>0.0984</td>
        <td>0.2547</td>
        <td>0.1788</td>
        <td>0.1676</td>
    </tr>
    <tr>
    <tr>
        <td rowspan=3>Generalization SC -> GA</td>
        <td>MAE</td>
        <td>29812.2109</td>
        <td>7731.0142</td>
        <td>5788.4355</td>
        <td>6306.5957</td>
        <td>7316.2769</td>
    </tr>
    <tr>
        <td>R<sup>2</td>
        <td>0.4087</td>
        <td>0.9602</td>
        <td>0.9809</td>
        <td>0.97</td>
        <td>0.9542</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.769</td>
        <td>0.1995</td>
        <td>0.1382</td>
        <td>0.1732</td>
        <td>0.214</td>
    </tr>
    <tr>
</table>




