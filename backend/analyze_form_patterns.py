import pdfplumber
import re
from pathlib import Path

def analyze_form_patterns():
    """PDF 파일에서 실제 서식 패턴들을 분석합니다."""
    
    # 분석할 파일들
    test_files = [
        "/app/documents/kisa_pdf/3_16_전자서명인증관리업무규칙(210809).pdf",
        "/app/documents/kisa_pdf/4_68_혁신경영 실행지침(230926).pdf",
        "/app/documents/kisa_pdf/4_69_대외 협력업무 처리지침(231228).pdf",
        "/app/documents/kisa_pdf/4_08_위탁과제 처리지침(191029).pdf",
        "/app/documents/kisa_pdf/4_32_정보보호 클러스터 운영에 관한 지침(170927).pdf",
        "/app/documents/kisa_pdf/3_11_계약사무처리규칙(240702).pdf"
    ]
    
    all_patterns = set()
    form_titles = set()
    
    for pdf_path in test_files:
        print(f"\n=== {Path(pdf_path).name} ===")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    lines = text.split('\n')
                    
                    # 첫 번째 라인에서 서식 패턴 찾기
                    if lines:
                        first_line = lines[0].strip()
                        
                        # 서식 시작 패턴 찾기
                        if re.search(r'\[별지\s*제\d+호\s*서식\]', first_line):
                            all_patterns.add('별지 제X호 서식')
                            print(f"  페이지 {page_num + 1}: {first_line}")
                            
                            # 제목 추출
                            for i in range(1, min(6, len(lines))):
                                line = lines[i].strip()
                                if line and len(line) > 2:
                                    form_titles.add(line)
                                    print(f"    제목: {line}")
                                    break
                        
                        elif re.search(r'\[별표\s*\d*\]', first_line):
                            all_patterns.add('별표 X')
                            print(f"  페이지 {page_num + 1}: {first_line}")
                            
                            # 제목 추출
                            for i in range(1, min(6, len(lines))):
                                line = lines[i].strip()
                                if line and len(line) > 2:
                                    form_titles.add(line)
                                    print(f"    제목: {line}")
                                    break
                        
                        elif re.search(r'\[부록\s*\d*\]', first_line):
                            all_patterns.add('부록 X')
                            print(f"  페이지 {page_num + 1}: {first_line}")
                        
                        elif re.search(r'\[첨부서식\s*\d*\]', first_line):
                            all_patterns.add('첨부서식 X')
                            print(f"  페이지 {page_num + 1}: {first_line}")
                        
                        elif re.search(r'\[첨부양식\s*\d*\]', first_line):
                            all_patterns.add('첨부양식 X')
                            print(f"  페이지 {page_num + 1}: {first_line}")
                        
        except Exception as e:
            print(f"  오류: {str(e)}")
    
    print(f"\n=== 발견된 패턴들 ===")
    for pattern in sorted(all_patterns):
        print(f"- {pattern}")
    
    print(f"\n=== 발견된 제목들 ===")
    for title in sorted(form_titles):
        print(f"- {title}")

if __name__ == "__main__":
    analyze_form_patterns()
