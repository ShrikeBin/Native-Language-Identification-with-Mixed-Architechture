# Architecture, Training and Results

## 1
- roberta embedding
- conv [512-1, 2048-3]
- max pool
- linear head
- accuracy: 72%
- **[BEST SO FAR]**

## 2
- own embedding [32]
- conv [512-1]
- max pool
- linear head
- accuracy: 52%

## 3
- own embedding [128]
- conv [512-1] (similar for bigger kernel size)
- max pool
- linear head
- accuracy: 59%

## 4
- own embedding [128]
- conv [512-3, 1024-3]
- max pool
- linear head
- accuracy: 58%

## 5
- own embedding [128]
- conv[1024-3]
- max pool
- linear head
- accuracy: 63%

## 6
- own embedding [512] (similar for bigger embedding)
- conv [1024-3]
- max pool
- linear head
- accuracy: 69%

## 7
- own embedding [512]
- conv [2048-3]
- max pool (similar for avg pool)
- linear head
- accuracy: 69%

## 8
- roberta embedding
- conv [256-2, 512-5]
- max pool
- linear head
- accuracy: 66%

## 9
- roberta embedding
- conv [512-2, 512-5]
- max pool
- linear head
- accuracy: 69%

## 10
- roberta embedding
- conv [512-3, 512-3]
- run conv2 on x step 3 in parallel
- max pool x2
- linear head
- accuracy: 66%

## 11
- roberta embedding
- conv [512-1, 512-3, 512-3]
- run conv2 on x step 3 in parallel
- max pool x2
- linear head
- accuracy: 64%
