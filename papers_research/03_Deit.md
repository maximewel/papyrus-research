* Paper:Training data-efficient image transformersm& distillation through attention
* Link: https://arxiv.org/pdf/2012.12877.pdf

Concept: Use a distillation token with a teacher-student mecanism in order to compress/train a smaller model from a highly trained one

-> We don't have any highly trained model that already extracts the time,x,y positions from the input image

Interesting: Written that ViT-type transformers are **very sensitive** to hyper-param (Batch size, learning rate, optimizer) **and** data augmentation as they require a lot of data to work.