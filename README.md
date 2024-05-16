<div align=center>## Elegantly Written: Disentangling Writer and Character Styles for Enhancing Online Chinese  Handwriting </div><hr>### 🛠️ Installation#### Prerequisites (Recommended)- Linux- Python 3.9- Pytorch 1.13.1- CUDA 12.2<hr>### Prepare handwriting data**Step 1**: Download DatasetCASIA-OLHWDB (1.0-1.2) download link```bashhttps://nlpr.ia.ac.cn/databases/handwriting/Home.html```The dataset consists of 1,020 files, each of which (*.pot) stores a sample of a human-written character.OLHWDB1.0: contains 3,866 Chinese characters and 171 alphanumeric characters and symbols. 3,740 of the 3,866 Chinese characters are in the GB2312-80 level 1 character set (3,755 characters in total).OLHWDB1.1: contains 3,755 GB2312-80 level 1 characters and 171 alphanumeric characters and symbols.OLHWDB1.2: contains 3,319 Chinese characters and 171 alphanumerics and symbols. the set of Chinese characters (3,319 classes) in OLHWDB1.2 is a disjoint set of OLHWDB1.0.OLHWDB1.0: and OLHWDB1.2 contain a total of 7,185 Chinese characters (7,185=3,866+3,319), which includes all 6,763 Chinese characters in GB2312.TestData: Based on the CASIA-HWDB and CASIA-OLHWDB databases. There are four datasets generated by 60 writers: offline character data, online character data, offline text data, online text data. The data format specifications can be found in the pages of Offline **Step 2**: Data ConstructionThe training data files tree should be :```DataPreparation├── POTData│   ├──001.pot│   ├──002.pot│   ├──003.pot│   └── ...├── SVGData│   ├──001.pot│   │  ├──\u4e8c.svg│   │  ├──\u4e9c.svg│   │  └── ...│   ├──002.pot│   │  ├──\u4e8c.svg│   │  ├──\u4e9c.svg│   │  └── ...│   ├──003.pot│   │  ├──\u4e8c.svg│   │  ├──\u4e9c.svg│   │  └── ...│   └── ...├── LMDB│   ├──lmdb│   │  ├──data.mdb│   │  └──lock.mdb```**Step 3**: Processing dataPlace the dataset downloaded in the first step in DataPreparation\POTData, and then run  **Pot2LMDB.py** .Running **readlmdb.ipynb** enables you to view the handwritten tracks stored by lmdb and convert them to SVG images for viewing.Running **Pot2SVG.py** converts the data in CASIA-OLHWDB (1.0-1.2) to SVG images for use in other projects. You can enter the characters you want to convert in **charlist.json**.<hr>### Content-Style Reference Mappingdecomposition.json  is the character structure decomposition table.  The first thing you need to do is to construct a content-style reference mapping table. ```bash{content1: [ref1, ref2, ref3, ...],content2: [ref1, ref2, ref3, ...],...}```example(in utf-8 format):```bash{研: [砍, 妍],级: [结, 没],脚: [胖, 法, 即], ...}```<hr>### train**Run scripts**```bashpython3 train.py ```## Citation```@inproceedings{yang2024fontdiffuser,  title={Elegantly Written: Disentangling Writer and Character Styles for Enhancing Online Chinese  Handwriting},  year={2024}}```