import React, { useState, useEffect, useRef } from 'react';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { exportSvgToImage } from '../utils/Exporter';

export default function BoardVisualizer({ parsedNodes }) {
  const svgRef = useRef(null);

  // 1. FILTER AND GROUP NODES
  const iga = parsedNodes.find(n => n.type.toUpperCase() === 'IGA') || { type: 'IGA', rating: '40A', poles: '2P' };
  const spd = parsedNodes.find(n => n.type.toUpperCase() === 'SPD') || { type: 'SPD' };
  const rcds = parsedNodes.filter(n => ['RCD', 'RCD_SI'].includes(n.type.toUpperCase()));
  const pias = parsedNodes.filter(n => ['PIA', 'MCB'].includes(n.type.toUpperCase()));

  // Ensure at least one differential group
  const effectiveRcds = rcds.length > 0 ? rcds : [{ id: 'default-rcd', type: 'RCD', rating: '40A', poles: '2P', sensitivity: '30mA' }];

  // Group PIAs under their parent differentials
  const rcdGroups = effectiveRcds.map(rcd => {
    let children = pias.filter(pia => pia.lineIndex === rcd.lineIndex);
    return { rcd, children };
  });

  // Assign any leftover PIAs (not inline on the same line as RCDs) to the first RCD group
  pias.forEach(pia => {
    const isAssigned = rcdGroups.some(g => g.children.some(c => c.id === pia.id));
    if (!isAssigned && rcdGroups.length > 0) {
      rcdGroups[0].children.push(pia);
    }
  });

  // 2. STATE FOR EDITABLE CIRCUIT CHARACTERISTICS (THE BOTTOM TABLE DATA)
  const [circuitData, setCircuitData] = useState({});

  useEffect(() => {
    setCircuitData(prev => {
      const updated = { ...prev };
      pias.forEach((pia, idx) => {
        if (!updated[pia.id]) {
          // Detect amperage rating from OCR/text
          const rawRating = (pia.rating || '').toUpperCase();
          let amp = '16A';
          if (rawRating.includes('10')) amp = '10A';
          else if (rawRating.includes('20')) amp = '20A';
          else if (rawRating.includes('25')) amp = '25A';
          else if (rawRating.includes('32')) amp = '32A';

          // Assign REBT defaults
          let usage = `Circuito C${idx + 1}`;
          let power = '3450';
          let section = '2.5';
          let tube = '20';

          if (amp === '10A') {
            usage = 'Alumbrado';
            power = '2300';
            section = '1.5';
            tube = '18';
          } else if (amp === '20A') {
            usage = 'Lavadora / Termo';
            power = '4600';
            section = '4.0';
            tube = '20';
          } else if (amp === '25A') {
            usage = 'Cocina y Horno';
            power = '5750';
            section = '6.0';
            tube = '25';
          } else if (amp === '32A') {
            usage = 'Cargador Vehículo Eléctrico';
            power = '7360';
            section = '10.0';
            tube = '32';
          } else if (idx === 0) {
            usage = 'Tomas de Uso General';
          } else if (idx === 1) {
            usage = 'Tomas Baño y Cocina Auxiliar';
          }

          updated[pia.id] = {
            label: pia.target || `C${idx + 1}`,
            usage,
            power,
            voltage: '230',
            section,
            tube
          };
        }
      });
      return updated;
    });
  }, [parsedNodes]);

  const handleCellChange = (id, field, value) => {
    setCircuitData(prev => ({
      ...prev,
      [id]: {
        ...prev[id],
        [field]: value
      }
    }));
  };

  // 3. LAYOUT MATHEMATICS
  const totalPiasCount = pias.length || 1;
  const colWidth = 160;
  const tableXStart = 150;
  const svgWidth = Math.max(1000, tableXStart + totalPiasCount * colWidth + 50);
  const svgHeight = 930;

  // Render SVG Symbol functions
  const drawConductorSlashes = (x, y) => (
    <g stroke="#000000" strokeWidth="1.5">
      <line x1={x - 6} y1={y - 3} x2={x + 6} y2={y + 3} />
      <line x1={x - 6} y1={y} x2={x + 6} y2={y + 6} />
    </g>
  );

  const drawGroundSymbol = (x, y) => (
    <g stroke="#000000" strokeWidth="1.5">
      <line x1={x} y1={y} x2={x} y2={y + 12} />
      <line x1={x - 10} y1={y + 12} x2={x + 10} y2={y + 12} />
      <line x1={x - 6} y1={y + 15} x2={x + 6} y2={y + 15} />
      <line x1={x - 2} y1={y + 18} x2={x + 2} y2={y + 18} />
    </g>
  );

  const triggerDownload = (format) => {
    if (svgRef.current) {
      exportSvgToImage(svgRef.current, 'esquema_unifilar', format, 2);
    }
  };

  const triggerPrint = () => {
    window.print();
  };

  return (
    <div className="visualizer-container">
      <div className="visualizer-header-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3 className="visualizer-title" style={{ margin: 0 }}>Esquema Unifilar Oficial (UNE-EN 60617)</h3>
        
        <div className="diagram-actions-row" style={{ display: 'flex', gap: '8px' }}>
          <button className="macro-btn main-head-btn" onClick={() => triggerDownload('png')}>
            Descargar PNG
          </button>
          <button className="macro-btn main-head-btn" style={{ background: '#3b82f6', color: '#fff', borderColor: '#2563eb' }} onClick={triggerPrint}>
            Imprimir / Guardar PDF
          </button>
        </div>
      </div>

      {pias.length === 0 ? (
        <div className="empty-visualizer-state">
          <p>Ningún interruptor detectado. Sube una foto en la pestaña de Análisis o introduce circuitos en el editor de la izquierda.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', background: '#1c1924', padding: '20px', borderRadius: '12px', border: '1px solid rgba(145, 55, 175, 0.2)' }}>
          
          {/* 4. INTERACTIVE SIDEBAR CONTROLS TO MANUALLY EDIT ATTRIBUTES */}
          <div className="diagram-editor-table" style={{ overflowX: 'auto', background: '#100e16', padding: '16px', borderRadius: '8px' }}>
            <span className="macro-label" style={{ marginBottom: '12px' }}>Datos Técnicos del Cuadro Eléctrico (Editar Valores):</span>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', color: '#e5e7eb', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  <th style={{ padding: '8px' }}>Ref</th>
                  <th style={{ padding: '8px' }}>Uso / Consumo</th>
                  <th style={{ padding: '8px' }}>Potencia (W)</th>
                  <th style={{ padding: '8px' }}>Tensión (V)</th>
                  <th style={{ padding: '8px' }}>Cable (mm²)</th>
                  <th style={{ padding: '8px' }}>Tubo (mm)</th>
                </tr>
              </thead>
              <tbody>
                {pias.map((pia) => {
                  const data = circuitData[pia.id] || { label: '', usage: '', power: '', voltage: '230', section: '1.5', tube: '16' };
                  return (
                    <tr key={pia.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '6px', fontWeight: 'bold', color: '#c084fc' }}>{data.label}</td>
                      <td style={{ padding: '6px' }}>
                        <input 
                          type="text" 
                          value={data.usage} 
                          onChange={(e) => handleCellChange(pia.id, 'usage', e.target.value)}
                          style={{ background: '#191722', border: '1px solid rgba(255,255,255,0.1)', color: '#fff', padding: '4px 8px', borderRadius: '4px', width: '100%' }}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <input 
                          type="number" 
                          value={data.power} 
                          onChange={(e) => handleCellChange(pia.id, 'power', e.target.value)}
                          style={{ background: '#191722', border: '1px solid rgba(255,255,255,0.1)', color: '#fff', padding: '4px 8px', borderRadius: '4px', width: '90px' }}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <input 
                          type="text" 
                          value={data.voltage} 
                          onChange={(e) => handleCellChange(pia.id, 'voltage', e.target.value)}
                          style={{ background: '#191722', border: '1px solid rgba(255,255,255,0.1)', color: '#fff', padding: '4px 8px', borderRadius: '4px', width: '60px' }}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <input 
                          type="text" 
                          value={data.section} 
                          onChange={(e) => handleCellChange(pia.id, 'section', e.target.value)}
                          style={{ background: '#191722', border: '1px solid rgba(255,255,255,0.1)', color: '#fff', padding: '4px 8px', borderRadius: '4px', width: '70px' }}
                        />
                      </td>
                      <td style={{ padding: '6px' }}>
                        <input 
                          type="text" 
                          value={data.tube} 
                          onChange={(e) => handleCellChange(pia.id, 'tube', e.target.value)}
                          style={{ background: '#191722', border: '1px solid rgba(255,255,255,0.1)', color: '#fff', padding: '4px 8px', borderRadius: '4px', width: '70px' }}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* 5. Crisp, High-Contrast SVG Drawing (Black/White for submission) */}
          <div style={{ background: '#ffffff', padding: '16px', borderRadius: '8px', boxShadow: 'inset 0 0 10px rgba(0,0,0,0.1)' }}>
            <TransformWrapper minScale={0.25} maxScale={8} initialScale={1} centerOnInit wheel={{ step: 0.12 }} doubleClick={{ mode: 'reset' }}>
              {({ zoomIn, zoomOut, resetTransform }) => (
                <>
                  <div className="diagram-zoom-controls" style={{ display: 'flex', gap: '8px', marginBottom: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <button className="macro-btn" onClick={() => zoomIn()}>＋ Acercar</button>
                    <button className="macro-btn" onClick={() => zoomOut()}>－ Alejar</button>
                    <button className="macro-btn" onClick={() => resetTransform()}>⟲ Restablecer</button>
                    <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>Rueda para zoom · arrastra para mover · doble clic para restablecer</span>
                  </div>
                  <TransformComponent wrapperStyle={{ width: '100%', height: '72vh', cursor: 'grab', background: '#ffffff', borderRadius: '6px', border: '1px solid rgba(0,0,0,0.08)' }}>
            <svg
              ref={svgRef}
              width={svgWidth}
              height={svgHeight}
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              style={{ display: 'block', background: '#ffffff', fontFamily: 'Courier, monospaced', userSelect: 'none' }}
            >
              {/* White background card block */}
              <rect width="100%" height="100%" fill="#ffffff" />

              {/* 5A. MAIN INCOMING HEAD (LGA / DI) */}
              <g stroke="#000000" strokeWidth="2" fill="none">
                {/* Feeder circle origin */}
                <circle cx={400} cy={30} r={4} fill="#000000" />
                <line x1={400} y1={30} x2={400} y2={90} />
              </g>
              {drawConductorSlashes(400, 50)}
              <text x={415} y={45} fontSize="11" fontWeight="bold" fill="#000000">D.I. (4x10) mm² Cu</text>
              <text x={415} y={60} fontSize="11" fill="#000000">RZI-K (AS)</text>
              <text x={415} y={75} fontSize="11" fill="#000000">BAJO TUBO=32mm</text>

              {/* IGA SWITCH */}
              <g stroke="#000000" strokeWidth="2" fill="none">
                {/* Arm and terminals */}
                <circle cx={400} cy={90} r={3} fill="#000000" />
                <circle cx={400} y={130} r={3} />
                <line x1={400} y1={90} x2={385} y2={125} />
                
                {/* Thermal-magnetic cross/release flags */}
                <line x1={394} y1={110} x2={388} y2={116} />
                <line x1={388} y1={110} x2={394} y2={116} />
                <rect x={377} y={105} width={6} height={6} />
                <line x1={383} y1={108} x2={388} y2={108} strokeDasharray="2 2" />
                
                <line x1={400} y1={130} x2={400} y2={180} />
              </g>
              <text x={415} y={105} fontSize="11" fontWeight="bold" fill="#000000">IGA GEN</text>
              <text x={415} y={120} fontSize="11" fill="#000000">{iga.rating || '40A'} / {iga.poles || '2P'}</text>

              {/* SPD (Overvoltage Protector) */}
              <g stroke="#000000" strokeWidth="2" fill="none">
                <line x1={400} y1={155} x2={500} y2={155} />
                <rect x={500} y={140} width={30} height={30} />
                <line x1={500} y1={170} x2={530} y2={140} />
                <line x1={515} y1={170} x2={515} y2={190} />
              </g>
              {drawGroundSymbol(515, 190)}
              <text x={540} y={158} fontSize="10" fill="#000000">SPD Protec.</text>
              <text x={540} y={170} fontSize="10" fill="#000000">Sobretensiones</text>

              {/* 5B. MAIN BUSBAR (Thick Horizontal Line) */}
              <line x1={80} y1={180} x2={svgWidth - 80} y2={180} stroke="#000000" strokeWidth="6" strokeLinecap="square" />

              {/* 5C. DIFFERENTIAL RCD GROUPS & SUB-BUSBARS */}
              {(() => {
                let piaIdx = 0;
                return rcdGroups.map((group, groupIdx) => {
                  const groupPiaCount = group.children.length;
                  if (groupPiaCount === 0) return null;

                  const startX = tableXStart + piaIdx * colWidth;
                  const endX = tableXStart + (piaIdx + groupPiaCount - 1) * colWidth;
                  const rcdX = (startX + endX) / 2;

                  // Move running offset
                  piaIdx += groupPiaCount;

                  const rcdLabel = group.rcd.class === 'RCD_SI' ? 'ID GEN SI' : 'ID GEN';
                  const sens = group.rcd.sensitivity || '30mA';
                  const rating = group.rcd.rating || '40A';
                  const rcdDetail = rating.includes(sens) ? rating : `${rating} / ${sens}`;

                  return (
                    <g key={groupIdx}>
                      {/* Drop from main busbar */}
                      <line x1={rcdX} y1={180} x2={rcdX} y2={220} stroke="#000000" strokeWidth="2" />

                      {/* RCD Switch Symbol */}
                      <g stroke="#000000" strokeWidth="2" fill="none">
                        <circle cx={rcdX} cy={220} r={3} fill="#000000" />
                        <circle cx={rcdX} cy={260} r={3} />
                        <line x1={rcdX} y1={220} x2={rcdX - 15} y2={255} />
                        
                        {/* Summation current transformer loop (standard RCD mark) */}
                        <ellipse cx={rcdX} cy={240} rx={12} ry={6} />
                        <path d={`M ${rcdX - 12} 240 C ${rcdX - 25} 240, ${rcdX - 25} 220, ${rcdX - 15} 225`} />
                        <rect x={rcdX - 27} y={218} width={6} height={6} />
                        
                        <line x1={rcdX} y1={260} x2={rcdX} y2={290} />
                      </g>
                      <text x={rcdX + 15} y={232} fontSize="11" fontWeight="bold" fill="#000000">{rcdLabel}</text>
                      <text x={rcdX + 15} y={247} fontSize="11" fill="#000000">{rcdDetail}</text>

                      {/* Sub-busbar */}
                      <line x1={startX} y1={290} x2={endX} y2={290} stroke="#000000" strokeWidth="4" />

                      {/* Render individual branch circuits under sub-busbar */}
                      {group.children.map((pia, localIdx) => {
                        const globalIdx = pias.indexOf(pia);
                        const cX = tableXStart + globalIdx * colWidth;
                        const data = circuitData[pia.id] || { label: '', usage: '', power: '', section: '1.5', tube: '16' };

                        return (
                          <g key={pia.id}>
                            {/* Line from sub-busbar */}
                            <line x1={cX} y1={290} x2={cX} y2={320} stroke="#000000" strokeWidth="2" />

                            {/* PIA Switch Symbol */}
                            <g stroke="#000000" strokeWidth="2" fill="none">
                              <circle cx={cX} cy={320} r={3} fill="#000000" />
                              <circle cx={cX} cy={360} r={3} />
                              <line x1={cX} y1={320} x2={cX - 15} y2={355} />
                              
                              {/* Thermal release box & cross flags */}
                              <line x1={cX - 11} y1={340} x2={cX - 5} y2={346} />
                              <line x1={cX - 5} y1={340} x2={cX - 11} y2={346} />
                              <rect x={cX - 23} y={335} width={6} height={6} />
                              <line x1={cX - 17} y1={338} x2={cX - 11} y2={338} strokeDasharray="2 2" />

                              <line x1={cX} y1={360} x2={cX} y2={600} />
                            </g>
                            {drawConductorSlashes(cX, 420)}
                            <text x={cX + 12} y={332} fontSize="11" fontWeight="bold" fill="#000000">PIA</text>
                            <text x={cX + 12} y={347} fontSize="11" fill="#000000">{pia.rating || '16A'}</text>

                            {/* Conductor properties vertical text */}
                            <text 
                              x={cX - 8} 
                              y={540} 
                              transform={`rotate(-90, ${cX - 8}, 540)`} 
                              fontSize="9.5" 
                              fill="#000000"
                            >
                              {`(2x${data.section})+TTx${data.section}mm²Cu`}
                            </text>
                            <text 
                              x={cX + 16} 
                              y={540} 
                              transform={`rotate(-90, ${cX + 16}, 540)`} 
                              fontSize="9.5" 
                              fill="#000000"
                            >
                              {`BAJO TUBO=Ø${data.tube}mm`}
                            </text>

                            {/* Arrow at feeder termination */}
                            <path d={`M ${cX - 4} 592 L ${cX} 600 L ${cX + 4} 592 Z`} fill="#000000" />
                          </g>
                        );
                      })}
                    </g>
                  );
                });
              })()}

              {/* 5D. CHARACTERISTICS DATA TABLE (CAJETÍN) */}
              {(() => {
                const tableY = 600;
                const rowH = 30;
                const labelW = 150;
                const rowTitles = [
                  'CONSUMOS',
                  'Pcal (W)',
                  'Un (V)',
                  'Sf (mm²)',
                  'Tubo (mm)',
                  'LONGITUD (m)'
                ];

                return (
                  <g stroke="#000000" strokeWidth="1.5">
                    {/* Horizontal lines */}
                    {rowTitles.map((_, rIdx) => (
                      <line 
                        key={rIdx} 
                        x1={10} 
                        y1={tableY + rIdx * rowH} 
                        x2={svgWidth - 10} 
                        y2={tableY + rIdx * rowH} 
                      />
                    ))}
                    {/* Bottom boundary line */}
                    <line 
                      x1={10} 
                      y1={tableY + rowTitles.length * rowH} 
                      x2={svgWidth - 10} 
                      y2={tableY + rowTitles.length * rowH} 
                    />

                    {/* Vertical grid lines */}
                    <line x1={10} y1={tableY} x2={10} y2={tableY + rowTitles.length * rowH} />
                    <line x1={labelW} y1={tableY} x2={labelW} y2={tableY + rowTitles.length * rowH} />
                    
                    {pias.map((_, pIdx) => {
                      const x = labelW + pIdx * colWidth;
                      return (
                        <line 
                          key={pIdx} 
                          x1={x} 
                          y1={tableY} 
                          x2={x} 
                          y2={tableY + rowTitles.length * rowH} 
                        />
                      );
                    })}
                    {/* Right boundary line */}
                    <line 
                      x1={svgWidth - 10} 
                      y1={tableY} 
                      x2={svgWidth - 10} 
                      y2={tableY + rowTitles.length * rowH} 
                    />

                    {/* Table row headers */}
                    {rowTitles.map((title, rIdx) => (
                      <text 
                        key={rIdx} 
                        x={20} 
                        y={tableY + rIdx * rowH + 20} 
                        fontSize="10" 
                        fontWeight="bold"
                        fill="#000000"
                        stroke="none"
                      >
                        {title}
                      </text>
                    ))}

                    {/* Table dynamic cell values */}
                    {pias.map((pia, pIdx) => {
                      const data = circuitData[pia.id] || { label: '', usage: '', power: '', voltage: '230', section: '1.5', tube: '16' };
                      const colX = labelW + pIdx * colWidth + colWidth / 2;

                      return (
                        <g key={pia.id} stroke="none" fill="#000000" fontSize="10" textAnchor="middle">
                          {/* Row 1: Usage label (truncated if long) */}
                          <text x={colX} y={tableY + 0 * rowH + 20}>
                            {data.usage.length > 20 ? data.usage.substring(0, 18) + '..' : data.usage}
                          </text>
                          {/* Row 2: Power */}
                          <text x={colX} y={tableY + 1 * rowH + 20}>{data.power} W</text>
                          {/* Row 3: Voltage */}
                          <text x={colX} y={tableY + 2 * rowH + 20}>{data.voltage} V</text>
                          {/* Row 4: Section */}
                          <text x={colX} y={tableY + 3 * rowH + 20}>{data.section} mm²</text>
                          {/* Row 5: Tube */}
                          <text x={colX} y={tableY + 4 * rowH + 20}>Ø{data.tube} mm</text>
                          {/* Row 6: Length default placeholder */}
                          <text x={colX} y={tableY + 5 * rowH + 20}>15 m</text>
                        </g>
                      );
                    })}
                  </g>
                );
              })()}
            </svg>
                  </TransformComponent>
                </>
              )}
            </TransformWrapper>
          </div>
        </div>
      )}
    </div>
  );
}

