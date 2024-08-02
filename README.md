# DARAM-ai-V3
### 머신러닝 기반 얼굴검출/분류 AI 모델 DARAM-V3 입니다!
## 에러 코드 목록
### ![#FF0000](https://via.placeholder.com/15/FF0000/000000?text=+) 10000번 에러
- **오류 메시지**: No images found in specified directory
- **설명**: 지정된 디렉토리에서 이미지를 찾을 수 없습니다. 디렉토리에 이미지 파일이 존재하는지 확인하십시오.

### ![#FF0000](https://via.placeholder.com/15/FF0000/000000?text=+) 10001번 에러
- **오류 메시지**: 지정된 경로를 찾을 수 없습니다 (FileNotFoundError)
- **설명**: 지정된 경로가 존재하지 않습니다. 디렉토리 경로가 올바른지 확인하십시오.

### ![#00FF00](https://via.placeholder.com/15/00FF00/000000?text=+) 10002번 경고
- **경고 메시지**: iCCP: known incorrect sRGB profile
- **설명**: PNG 파일에 잘못된 sRGB 프로파일이 포함되어 있습니다. 이는 이미지 로딩에는 큰 영향을 미치지 않습니다.

### ![#FFFF00](https://via.placeholder.com/15/FFFF00/000000?text=+) 10003번 경고
- **경고 메시지**: cv::findDecoder imread_: can't open/read file: check file path/integrity
- **설명**: OpenCV에서 이미지를 로드할 때 파일을 찾거나 읽을 수 없습니다. 파일 경로가 정확한지 확인하십시오.