# DCN-FSA-Net-
提出DCN-FSA-Net，移动端中文车牌识别新框架：DCNv3几何适应定理强化边缘提取；FSA把自注意力复杂度从O(n²d)削至O(nd)；LP-Prior损失融入车牌先验，优化CTC解码。4.2 M参数、2.3 G FLOPs@416×416，Jetson Orin Nano跑67 FPS，E2E-mAP 83.7%；辅以字符分割，端到端准确率升至87.2%，字符分割98.9%，全面领先。
