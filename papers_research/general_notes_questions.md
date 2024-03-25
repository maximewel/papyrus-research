Vit are very good and SWIN iterations resolve the problem of "Transformers are quadratic repsective to the image size" (With self-attention being a matmult between the patches themselves, repeated multiple times during the process). Are we going to want to work on great images ? It appears to be better to already segment-out "lines" of text in case of papyrus, is it possible/done yet ? Carefull about Dructure flow.

Repeated multiple times: Transformers need lot of data to work, is dataset going to be great enoug ? Pre-training ? Can we pre-train on different dructus ?
    - Data augmentation ? (See DeiT)

Curious: Swin does not use positional embeddings, but positional cues in the self-attention computation.


Database not selected: https://nlpr.ia.ac.cn/databases/handwriting/Online_database.html
    Appears to be "stroke x,y" and not sampled with a frequency, which means we can't really extract the time well.