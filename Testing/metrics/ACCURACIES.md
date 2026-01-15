# Model Accuracies

| Task                              | Model / Variant                  | Accuracy (%) |
|-----------------------------------|----------------------------------|--------------|
| **Gender (class)**                | DistilBERT - Full Fine-tuned     | *76,4%*      |
| **Gender (reg)**                  | DistilBERT - Baseline            | *66,2%*      |
|                                   | DistilBERT - LoRA                | *72,6%*      |
|                                   | DistilBERT - Full Fine-tuned     | *76,5%*      |
|                                   | RoBERTa - Full Fine-tuned        | *77,3%*      |
|-----------------------------------|----------------------------------|--------------|
| **MBTI (class)**                  | DistilBERT - Baseline            | *22,3%*      |
|                                   | DistilBERT - LoRA                | *32,1%*      |
|                                   | DistilBERT - Full Fine-tuned     | *40,9%*      |
|-----------------------------------|----------------------------------|--------------|
| **Political (reg)**               | DistilBERT - Full Fine-tuned     | *78,7%*      |
|                                   | DistilBERT - LoRA                | *58,0%*      |
|                                   | DistilBERT - Baseline            | *22,8%*      |
| **Political (removed head)**      | DistilBERT - Full Fine-tuned     | *77,5%*      |
|-----------------------------------|----------------------------------|--------------|
| **Language (class)**              | DistilBERT - Baseline            | *46,5%*      |
|                                   | DistilBERT - LoRA                | *62,1%*      |
|                                   | DistilBERT - Full Fine-tuned     | *69,2%*      |
|                                   | DeBERTa-v3-Large - LoRA          | *70,5%*      |
|                                   | MPNet-v2 - Full Fine-tuned       | *68,7%*      |
|                                   | RoBERTa - Full Fine-tuned        | *74,1%*      |
|                                   | RoBERTa-Large - Full Fine-tuned  | *79,9%*      |
|                                   | CNN on RoBERTa embedder          | *73,6%*      |
|                                   | CNN + RoBERTa Classifier         | *77,3%*      |
|                                   | CNN + RoBERTaLarge Classofier    | *81,0%*      |
|-----------------------------------|----------------------------------|--------------|


| Task                              | Model / Variant                  | sqrt(MSE)    |
|-----------------------------------|----------------------------------|--------------|
| **Age (reg)**                     | DistilBERT - Baseline            | *7*          |
|                                   | DistilBERT - LoRA                | *6.25*       |
|                                   | DistilBERT - Full Fine-tuned     | *5.59*       |
|-----------------------------------|----------------------------------|--------------|
