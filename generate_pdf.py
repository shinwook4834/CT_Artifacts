import os
import subprocess

html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Research Proposal - Chemo Port MAR & Dosimetry</title>
<style>
  @page {
    size: A4 portrait;
    margin: 12mm 15mm 12mm 15mm;
  }
  
  * {
    box-sizing: border-box;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif;
    color: #1e293b;
    line-height: 1.45;
    font-size: 9.3pt;
    margin: 0;
    padding: 0;
    background-color: #ffffff;
  }

  .page {
    page-break-after: always;
    min-height: 268mm;
    max-height: 268mm;
    position: relative;
    padding-bottom: 25px;
    overflow: hidden;
  }

  .page:last-child {
    page-break-after: avoid;
  }

  /* Header & Footer */
  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1.5px solid #0f2942;
    padding-bottom: 5px;
    margin-bottom: 12px;
  }

  .header-badge {
    background: #0f2942;
    color: #ffffff;
    font-size: 7.2pt;
    font-weight: 700;
    padding: 2.5px 8px;
    border-radius: 3px;
    letter-spacing: 0.5px;
  }

  .header-meta {
    font-size: 7.5pt;
    color: #64748b;
    font-weight: 500;
  }

  .page-footer {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #cbd5e1;
    padding-top: 5px;
    font-size: 7.5pt;
    color: #94a3b8;
  }

  /* Typography */
  h1.doc-title {
    font-size: 14.8pt;
    font-weight: 800;
    color: #0f2942;
    line-height: 1.25;
    margin: 0 0 4px 0;
    letter-spacing: -0.3px;
  }

  .doc-subtitle {
    font-size: 9pt;
    font-style: italic;
    color: #475569;
    margin: 0 0 10px 0;
    line-height: 1.3;
  }

  h2.section-title {
    font-size: 11pt;
    font-weight: 700;
    color: #0f2942;
    border-left: 4px solid #1d4ed8;
    padding-left: 8px;
    margin: 10px 0 6px 0;
    display: flex;
    align-items: center;
  }

  h3.sub-title {
    font-size: 9.6pt;
    font-weight: 700;
    color: #1d4ed8;
    margin: 7px 0 3px 0;
  }

  p {
    margin: 0 0 5px 0;
    text-align: justify;
  }

  /* Cards & Callouts */
  .meta-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 10px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .meta-item {
    font-size: 8pt;
  }
  .meta-label {
    font-weight: 700;
    color: #64748b;
    display: block;
    margin-bottom: 1px;
    text-transform: uppercase;
    font-size: 7pt;
  }
  .meta-value {
    color: #0f2942;
    font-weight: 600;
  }

  .callout-box {
    background: #f0fdfa;
    border-left: 4px solid #0d9488;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin: 7px 0;
  }
  .callout-title {
    font-weight: 700;
    color: #0f766e;
    font-size: 8.8pt;
    margin-bottom: 2px;
  }
  .callout-text {
    font-size: 8.7pt;
    color: #134e4a;
    margin: 0;
    line-height: 1.35;
  }

  .alert-box {
    background: #fff1f2;
    border-left: 4px solid #e11d48;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin: 7px 0;
  }
  .alert-title {
    font-weight: 700;
    color: #be123c;
    font-size: 8.8pt;
    margin-bottom: 2px;
  }
  .alert-text {
    font-size: 8.7pt;
    color: #881337;
    margin: 0;
    line-height: 1.35;
  }

  /* Grid Layouts */
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 9px;
    margin: 6px 0;
  }

  .info-card {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 10px;
  }
  .info-card-header {
    font-weight: 700;
    font-size: 8.8pt;
    color: #0f2942;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .info-card-body {
    font-size: 8.2pt;
    color: #334155;
    line-height: 1.35;
  }

  /* Tables */
  table.custom-table {
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0 8px 0;
    font-size: 8pt;
  }
  table.custom-table th {
    background: #0f2942;
    color: #ffffff;
    font-weight: 700;
    padding: 5px 7px;
    text-align: left;
    border: 1px solid #0f2942;
  }
  table.custom-table td {
    padding: 4.5px 7px;
    border: 1px solid #cbd5e1;
    line-height: 1.3;
  }
  table.custom-table tr:nth-child(even) td {
    background: #f8fafc;
  }

  .tag {
    display: inline-block;
    padding: 1.5px 5px;
    border-radius: 3px;
    font-size: 7.2pt;
    font-weight: 700;
  }
  .tag-blue { background: #dbeafe; color: #1e40af; }
  .tag-teal { background: #ccfbf1; color: #0f766e; }
  .tag-rose { background: #ffe4e6; color: #be123c; }
  .tag-amber { background: #fef3c7; color: #92400e; }
  .tag-purple { background: #f3e8ff; color: #6b21a8; }

  ul {
    margin: 2px 0 6px 0;
    padding-left: 16px;
  }
  li {
    margin-bottom: 2.5px;
    font-size: 8.8pt;
    line-height: 1.35;
  }
</style>
</head>
<body>

<!-- ================= PAGE 1 ================= -->
<div class="page">
  <div class="page-header">
    <span class="header-badge">RESEARCH PROTOCOL & ROADMAP</span>
    <span class="header-meta">Medical Physics & Radiation Oncology Study</span>
  </div>

  <h1 class="doc-title">기치료 유방암 환자의 실제 치료계획 기반 케모포트 AI 인공음영 저감(MAR) 및 후향적 선량 재평가 연구</h1>
  <div class="doc-subtitle">Retrospective Dosimetric Evaluation of Breast Cancer Radiation Therapy with a Novel CAD Prior-Guided AI Metal Artifact Reduction for Patients with Chemo Ports</div>

  <div class="meta-card">
    <div class="meta-item">
      <span class="meta-label">연구 디자인</span>
      <span class="meta-value">후향적 임상-의학물리 융합 연구</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">목표 저널군</span>
      <span class="meta-value">Medical Physics, Green/Red Journal, PMB</span>
    </div>
    <div class="meta-item">
      <span class="meta-label">대상 코호트</span>
      <span class="meta-value">케모포트 유지 유방암 환자 (SCN/흉벽 RT)</span>
    </div>
  </div>

  <h2 class="section-title">1. 연구 배경 및 임상적 문제의식 (Clinical Rationale)</h2>
  <p>
    <strong>• 임상적 현주소:</strong> 전 세계 유방암 환자의 항암치료에 사용되는 케모포트의 약 60%는 여전히 고밀도 금속인 <strong>티타늄(Titanium)</strong> 재질입니다. 방사선 치료 시 케모포트가 쇄골상부 림프절(SCN) 및 흉벽 치료 조사야 내에 위치하면서 심각한 금속 인공음영(Metal Artifact)을 발생시킵니다.
  </p>
  <p>
    <strong>• 기존 임상 치료의 구조적 한계:</strong> 
    병원에서 사용 중인 상용 MAR(iMAR, O-MAR)은 케모포트처럼 <em>'피부 표면 바로 아래(Subcutaneous)'</em>이자 <em>'폐(공기) 경계면'</em>에 인접한 부위에서 심각한 경계 뭉개짐(Blurring)과 2차 암영을 유발합니다. 또한 포트 내부의 비금속 물질(실리콘 셉텀, 유체 챔버)을 무시한 채 전체를 통짜 금속이나 물로 강제 지정(Bulk Override)하여 치료계획을 수립해 왔습니다.
  </p>

  <div class="callout-box">
    <div class="callout-title">핵심 임상 질문 (Core Clinical Question)</div>
    <div class="callout-text">
      “우리가 과거에 이미 치료를 마친 환자들의 실제 몸속에는 선량이 얼마나 정확하게 전달되었던 것일까? 
      아티팩트로 인해 의사가 타깃(CTV/PTV)을 불완전하게 그렸거나, 포트 후방 림프절에 심각한 선량 결손(Cold spot), 또는 피부에 예측치 못한 과선량(Hot spot)이 발생하지는 않았는가?”
    </div>
  </div>

  <h2 class="section-title">2. 연구 핵심 가설 (Hypotheses)</h2>
  <div class="two-col">
    <div class="info-card">
      <div class="info-card-header"><span class="tag tag-blue">가설 1</span> 물리적 선량 계산 왜곡 검증</div>
      <div class="info-card-body">
        제조사 규격 기반 3D CAD Prior를 통해 '진짜 해부학 및 포트 내부 밀도'가 복원된 CT 위에 기존 승인 치료 빔(Beam, MU, Segment)을 동결 재계산(Frozen Recalculation)하면, TPS가 예측했던 선량과 임상적으로 유의미한 표적 결손 및 피부 과선량이 실증될 것이다.
      </div>
    </div>
    <div class="info-card">
      <div class="info-card-header"><span class="tag tag-rose">가설 2</span> 표적 누락(Geographic Miss) 규명</div>
      <div class="info-card-body">
        아티팩트가 쇄골하혈관 및 림프절 경계를 가려 치료 당시 의사가 타깃(CTV)을 왜곡되게 설정했을 것이며, 깨끗이 복원된 CT에서 재설정한 '진짜 표적(CTV_True)'에 기존 치료계획을 비추었을 때 표적 누락(Under-coverage)이 발생했음을 입증할 것이다.
      </div>
    </div>
  </div>

  <h2 class="section-title">3. 연구의 3대 핵심 독창성 (Key Novelties)</h2>
  <ul>
    <li><strong>제조사 공식 레퍼런스 규격 기반 다중 물질 3D CAD Prior:</strong> 고가의 마이크로 CT 없이 공식 사양서(BD PowerPort 등)를 바탕으로 외벽 티타늄, 실리콘 셉텀, 유체 챔버를 완벽 분리 모델링하여 오픈 사이언스 및 재현성 극대화.</li>
    <li><strong>Prior-guided 2-Stage Deep Learning Framework:</strong> 6-DoF 자세 추정으로 포트 내부는 100% 물리 밀도로 고정하고, 외벽과 정상 인체 경계를 조건으로 부여하는 조건부 인페인팅(Diffusion/Transformer)으로 해부학 복원.</li>
    <li><strong>치료 빔 동결 후향적 선량 평가 (Frozen-Plan Recalculation):</strong> 가상 팬텀에 그치지 않고 실제 병원에서 조사된 치료 플랜을 그대로 재계산하여 의료 현장에 즉각적인 경각심과 표준 가이드라인 제공.</li>
  </ul>

  <div class="page-footer">
    <span>Chemo Port MAR & Dosimetry Protocol</span>
    <span>Page 1 of 4 | Background & Clinical Rationale</span>
  </div>
</div>

<!-- ================= PAGE 2 ================= -->
<div class="page">
  <div class="page-header">
    <span class="header-badge">MATERIALS & METHODS</span>
    <span class="header-meta">CAD Prior & 2-Stage AI Architecture</span>
  </div>

  <h2 class="section-title">2. 연구 재료 및 방법 (Materials and Methods)</h2>

  <h3 class="sub-title">2.1 후향적 환자 코호트 및 DICOM-RT 데이터셋</h3>
  <p>
    본 연구는 쇄골상부 림프절(SCN) 및 흉벽을 포함하여 방사선 치료(VMAT 또는 3D-CRT/IMRT)를 완료한 케모포트 삽입 유방암 환자 코호트(20~30례)를 대상으로 합니다. 수집 데이터는 <strong>원본 모의치료 CT, 상용 MAR 적용 CT, 승인된 치료계획(RP/Plan), 윤곽 구조물(RS/Structure), 계산된 3차원 선량 분포(RD/Dose)</strong>입니다.
  </p>

  <h3 class="sub-title">2.2 제조사 레퍼런스 규격 기반 다중 물질(Multi-material) 3D CAD Prior</h3>
  <p>
    케모포트 내부의 밀도 불균질성을 반영하기 위해, 제조사 공식 규격을 기반으로 3개 파트로 분리된 파라메트릭 CAD 모델을 구축합니다. 체적(0.6 mL) 및 총중량(7.5 g) 자체 검증을 통해 오차를 원천 차단합니다.
  </p>

  <table class="custom-table">
    <thead>
      <tr>
        <th style="width: 28%;">구분 (Compartment)</th>
        <th style="width: 38%;">물리적 재질 및 전자밀도 (Density)</th>
        <th style="width: 34%;">방사선 치료학적 의미</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Part 1: 외벽 및 바닥 챔버</strong></td>
        <td>Titanium Alloy (ρ = 4.43 g/cm³, HU > 7000)</td>
        <td>방사선 차폐(Cold spot) 및 경계면 전자 후방산란 유발</td>
      </tr>
      <tr>
        <td><strong>Part 2: 실리콘 격막 (Septum)</strong></td>
        <td>Silicone Elastomer (ρ = 1.15 g/cm³, HU ~ 120)</td>
        <td>연부조직과 유사 투과 특성 (통짜 금속 오판 시 선량 과대평가)</td>
      </tr>
      <tr>
        <td><strong>Part 3: 내부 유체 챔버</strong></td>
        <td>Fluid Cavity (ρ = 1.00 g/cm³, HU ~ 0)</td>
        <td>생리식염수 충만 상태 모사 (광자선이 자유롭게 통과)</td>
      </tr>
    </tbody>
  </table>

  <h3 class="sub-title">2.3 2-Stage Prior-Guided AI 복원 파이프라인</h3>
  <div class="two-col">
    <div class="info-card" style="border-top: 3px solid #1d4ed8;">
      <div class="info-card-header"><span class="tag tag-blue">Stage 1</span> 6-DoF 자세 추정 & CAD 정합</div>
      <div class="info-card-body">
        • <strong>네트워크:</strong> 3D CNN 기반의 공간 자세 추귀(Pose Regression) 네트워크.<br>
        • <strong>동작 원리:</strong> 아티팩트로 번진 CT 속에서 포트의 중심점(x, y, z)과 회전각(Roll, Pitch, Yaw)을 정밀 산출.<br>
        • <strong>결과:</strong> 규격화된 CAD 모델을 오버레이하여 <em>진짜 금속 외벽 경계를 100% 확정</em>하고, 내부는 실제 밀도(티타늄/실리콘/유체)로 고정.
      </div>
    </div>
    <div class="info-card" style="border-top: 3px solid #0d9488;">
      <div class="info-card-header"><span class="tag tag-teal">Stage 2</span> 조건부 인페인팅 (Diffusion/Transformer)</div>
      <div class="info-card-body">
        • <strong>네트워크:</strong> Conditional Latent Diffusion 또는 Dual-Domain Swin-Transformer.<br>
        • <strong>조건 주입:</strong> 진짜 포트 외벽(Inner boundary) + 줄무늬 밖 정상 피부선/폐/늑골(Outer boundary).<br>
        • <strong>결과:</strong> 상용 MAR이 뭉개뜨리던 쇄골하혈관, 림프절 구획(SCN), 피부 경계선을 <em>인체 해부학적 연속성에 맞게 완벽 복원</em>.
      </div>
    </div>
  </div>

  <h3 class="sub-title">2.4 동결 플랜 선량 재계산 방법론 (Frozen-Plan Recalculation)</h3>
  <p>
    환자의 기존 치료계획에 포함된 모든 물리적 빔 파라미터(빔 각도, 세그먼트 형태, 모니터 유닛[MU], 갠트리 궤적 등)를 <strong>100% 동결(Freeze)</strong>한 채, 다음 3가지 CT 볼륨에 선량만 순방향 재계산(Forward calculation)합니다:
  </p>
  <ul>
    <li><strong>Plan_Orig :</strong> 치료 당시 승인된 기존 오리지널 CT 기반 계산 선량</li>
    <li><strong>Plan_Comm :</strong> 병원 상용 MAR(iMAR/O-MAR) 적용 CT 기반 계산 선량</li>
    <li><strong>Plan_AI :</strong> 본 연구의 다중 물질 CAD Prior AI 복원 CT 기반 선량 <em>(실제 환자 몸에 들어간 참 선량의 기준)</em></li>
    <li><em>(검증군)</em> Geant4 몬테카를로 시뮬레이션을 통한 물리적 정밀 벤치마크 선량 병행 산출</li>
  </ul>

  <div class="page-footer">
    <span>Chemo Port MAR & Dosimetry Protocol</span>
    <span>Page 2 of 4 | Materials & Methods</span>
  </div>
</div>

<!-- ================= PAGE 3 ================= -->
<div class="page">
  <div class="page-header">
    <span class="header-badge">EVALUATION & ENDPOINTS</span>
    <span class="header-meta">Two-Track Target Analysis & Expected Impact</span>
  </div>

  <h2 class="section-title">3. 정량적 평가 지표 (Quantitative Endpoints)</h2>

  <h3 class="sub-title">3.1 Two-Track 표적 평가 체계 (Target Evaluation Framework)</h3>
  <p>
    아티팩트로 인해 혈관과 해부학적 경계가 가려져 치료 당시 타깃(CTV/PTV) 자체가 왜곡되었을 가능성을 고려하여, <strong>기하학적 오차(Track A)</strong>와 <strong>선량학적 오차(Track B)</strong>로 분리 평가합니다.
  </p>

  <table class="custom-table">
    <thead>
      <tr>
        <th style="width: 30%;">평가 트랙 (Evaluation Track)</th>
        <th style="width: 42%;">세부 분석 항목 및 정량 지표</th>
        <th style="width: 28%;">임상적 핵심 가치</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>
          <span class="tag tag-blue">Track A</span><br>
          <strong>기하학적 윤곽 오차</strong><br>
          (Contouring Geometric Error)
        </td>
        <td>
          • 복원된 AI-CT에서 전문의가 가이드라인(ESTRO/RTOG)에 따라 <strong>'진짜 표적(CTV_True)'</strong> 재설정<br>
          • <strong>지표:</strong> 체적 차이(ΔV), 일치도(Dice Similarity Coefficient, DSC), 95% Hausdorff Distance (HD95)
        </td>
        <td>
          아티팩트로 인해 치료 당시 의사가 타깃을 얼마나 과소/과대 설정했는가 정량 입증
        </td>
      </tr>
      <tr>
        <td>
          <span class="tag tag-rose">Track B-1</span><br>
          <strong>순수 물리적 선량 오차</strong><br>
          (on Frozen CTV_Orig)
        </td>
        <td>
          • 기존 타깃 윤곽(CTV_Orig)을 그대로 동결한 채 선량만 비교<br>
          • <strong>지표:</strong> D95%, Dmean, 포트 직후방 Cold spot 부피 및 선량 저하율(ΔD)
        </td>
        <td>
          의사가 의도했던 영역에 물리적으로 선량이 제대로 전달되었는가 검증
        </td>
      </tr>
      <tr>
        <td>
          <span class="tag tag-purple">Track B-2</span><br>
          <strong>임상적 표적 누락 분석</strong><br>
          (on Re-contoured CTV_True) ★
        </td>
        <td>
          • <strong>핵심 질문:</strong> “기존 치료계획(Plan_Orig)은 새로 밝혀진 '진짜 림프절 표적(CTV_True)'을 얼마나 놓쳤는가?”<br>
          • <strong>지표:</strong> V95%(CTV_True), D98%, PTV coverage 결손율 산출
        </td>
        <td>
          <strong>논문의 최고 킬러 포인트:</strong><br>
          타깃 윤곽 오류로 인한 실제 암 조직의 방사선 미조사(Geographic Miss) 규명
        </td>
      </tr>
    </tbody>
  </table>

  <h3 class="sub-title">3.2 정상장기(OAR) 및 피부 선량 평가</h3>
  <div class="two-col">
    <div class="info-card">
      <div class="info-card-header"><span class="tag tag-amber">피부 선량</span> Skin Toxicity Risk</div>
      <div class="info-card-body">
        • 포트 직상방 피부의 최대선량(Dmax) 및 평균선량(Dmean).<br>
        • 금속 경계면 전자 후방산란으로 인해 당초 계산보다 피부에 들어간 과선량(Hot spot) 및 피부염 위험도 정량화.
      </div>
    </div>
    <div class="info-card">
      <div class="info-card-header"><span class="tag tag-teal">폐 선량 & 3D 감마</span> Organ at Risk & Gamma</div>
      <div class="info-card-body">
        • 동측 폐 V20Gy, V5Gy 평가 (상용 MAR 계면 블러로 인한 왜곡 폭 규명).<br>
        • 3D Gamma Pass Rate (3%/3mm, 2%/2mm, 1%/1mm) 전역 일치도 검증.
      </div>
    </div>
  </div>

  <h2 class="section-title">4. 예상 결과 및 임상적 기대효과 (Expected Impact)</h2>
  <ul>
    <li><strong>미지의 임상적 사실 규명:</strong> 과거 치료계획 시스템이 케모포트 아티팩트 및 비금속 내부 물질 무시로 인해 림프절 표적에 얼마만큼의 선량 결손을 유발했는지 세계 최초로 정량 보고.</li>
    <li><strong>표적 누락(Geographic Miss) 위험성 경고:</strong> 아티팩트로 인해 의사가 타깃을 불완전하게 설정함으로써 발생했던 잠재적 종양 미조사(Under-dosage) 위험을 입증하여 방사선종양학 진료 지침 개선에 기여.</li>
    <li><strong>임상 워크플로우 혁신:</strong> 번거롭고 주관적인 수동 밀도 보정 없이, AI 기반의 자동화된 초정밀 CT 복원 및 선량 계산 표준 프로토콜 확립.</li>
  </ul>

  <div class="page-footer">
    <span>Chemo Port MAR & Dosimetry Protocol</span>
    <span>Page 3 of 4 | Evaluation & Expected Impact</span>
  </div>
</div>

<!-- ================= PAGE 4 ================= -->
<div class="page">
  <div class="page-header">
    <span class="header-badge">THE GRAND VISION</span>
    <span class="header-meta">5-Stage Comprehensive Artifact Reduction Roadmap</span>
  </div>

  <h2 class="section-title">5. 방사선 치료 체내 임플란트 아티팩트 극복 5대 핵심 로드맵</h2>
  <p>
    본 케모포트 연구는 인체 내 고밀도 의료기기로 인해 발생하는 방사선 치료 오차를 완전 정복하기 위한 <strong>5부작 연속 연구(The 5-Stage Grand Vision)</strong>의 첫 번째 출발점입니다.
  </p>

  <table class="custom-table" style="font-size: 7.7pt;">
    <thead>
      <tr>
        <th style="width: 14%;">단계</th>
        <th style="width: 22%;">대상 임플란트 및 부위</th>
        <th style="width: 28%;">주변 고위험 구조물 및 문제점</th>
        <th style="width: 36%;">핵심 연구 질문 및 차별화 AI 해법</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background: #f0fdf4;">
        <td><strong>제1탄<br>(본 연구)</strong></td>
        <td><strong>케모포트 (Chemo Port)</strong><br><span class="tag tag-teal">유방암 / 쇄골상부</span></td>
        <td>피부 표면 바로 밑, 폐 계면 왜곡, 쇄골상부 림프절 가림</td>
        <td><strong>다중 물질 CAD Prior + 2-Stage AI</strong><br>후향적 선량 재계산 및 타깃 누락(Geographic Miss) 실증</td>
      </tr>
      <tr>
        <td><strong>제2탄<br>★신규 확장</strong></td>
        <td><strong>척추경 나사 (Spine Screws)</strong><br><span class="tag tag-rose">척추 전이암 SBRT</span></td>
        <td><strong>1~2mm 인접 척수(Spinal Cord)</strong><br>과선량 시 하반신 마비 영구 장애 유발</td>
        <td><strong>나사산(Thread) CAD Prior + 척수 PRV 초정밀 복원</strong><br>척수 마진 침범 방지 및 급격한 선량 감쇄(Gradient) 정밀화</td>
      </tr>
      <tr>
        <td><strong>제3탄<br>★신규 확장</strong></td>
        <td><strong>인공 심박동기 (Pacemaker/ICD)</strong><br><span class="tag tag-amber">폐암 / 좌측 유방암</span></td>
        <td><strong>2 Gy 피폭 한계 반도체 칩</strong><br>금속 배터리 팩에 의한 선량 왜곡으로 기기 소손</td>
        <td><strong>금속 배터리 하우징 CAD 정합 + 기기 피폭 선량 정밀화</strong><br>기기 오작동 사망 방지 및 주변 폐 종양 선량 동시 확보</td>
      </tr>
      <tr>
        <td><strong>제4탄</strong></td>
        <td><strong>치아 보철물 (Dental Implants)</strong><br><span class="tag tag-blue">두경부암 / 구강·인후두</span></td>
        <td>다발성 금속체 상호 교차 줄무늬<br>침샘(구강건조증), 연하근(연하장애) 파괴</td>
        <td><strong>다중 클러스터 Sinogram-Image Dual Domain AI</strong><br>다방향 줄무늬 분리 억제 및 정상 침샘/하악골 해부학 복원</td>
      </tr>
      <tr>
        <td><strong>제5탄</strong></td>
        <td><strong>인공 고관절 (Hip Prosthesis)</strong><br><span class="tag tag-purple">골반암 / 전립선·부인과</span></td>
        <td><strong>초대형 완전 광자 결핍 (Total Starvation)</strong><br>골반 심부 방광, 직장, 전립선 선량 붕괴</td>
        <td><strong>대형 인공관절 CAD Prior + 몬테카를로 물리 통합 AI</strong><br>검은 암영(Black hole) 영역 복원 및 골반 표적 선량 정상화</td>
      </tr>
    </tbody>
  </table>

  <h2 class="section-title">6. 연구 실행 로드맵 및 향후 추진 계획 (Execution Plan)</h2>
  <div class="two-col">
    <div class="info-card">
      <div class="info-card-header"><span class="tag tag-blue">Phase 1 (단기 목표)</span> 케모포트 논문 완성</div>
      <div class="info-card-body">
        • BD PowerPort 레퍼런스 규격 CAD 모델링 및 부피/중량 검증.<br>
        • 가상 시뮬레이션 페어 데이터셋 구축 및 2-Stage AI 학습.<br>
        • 기치료 유방암 환자 코호트 선량 재계산 및 논문 투고 (Med Phys).
      </div>
    </div>
    <div class="info-card">
      <div class="info-card-header"><span class="tag tag-teal">Phase 2 (중장기 목표)</span> 척추 SBRT & 기기 스핀오프</div>
      <div class="info-card-body">
        • 척추경 나사 CAD 라이브러리 확장 및 척수 마비 예방 임상 연구 착수.<br>
        • 페이스메이커 2 Gy 안전 마진 보장을 위한 기기 특화 MAR 파이프라인 개발.<br>
        • 5부작 시리즈 완성을 통한 글로벌 방사선 치료 MAR 분야 선도.
      </div>
    </div>
  </div>

  <div class="page-footer">
    <span>Chemo Port MAR & Dosimetry Protocol</span>
    <span>Page 4 of 4 | The 5-Stage Grand Vision Roadmap</span>
  </div>
</div>

</body>
</html>
"""

html_path = "/Users/shinwookkim/Documents/Geant4/ChemoPort_MAR_Research_Proposal.html"
pdf_path = "/Users/shinwookkim/Documents/Geant4/ChemoPort_MAR_Research_Proposal.pdf"

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cmd = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    html_path
]

res = subprocess.run(cmd, capture_output=True, text=True)
print("Chrome return code:", res.returncode)
if os.path.exists(pdf_path):
    size = os.path.getsize(pdf_path)
    print(f"Successfully generated {pdf_path} ({size} bytes)")
else:
    print("Failed to generate PDF")
