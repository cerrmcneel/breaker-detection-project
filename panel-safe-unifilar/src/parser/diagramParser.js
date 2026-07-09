/**
 * Parser for the token-efficient unifilar board layout code.
 * 
 * Syntax:
 * - Breaker components: [TYPE:PARAM1:PARAM2...] (e.g. [IGA:40A:2P], [PIA:16A:2.5mm:T20])
 * - Target circuits: Plain text (e.g. C1, C2, "Alumbrado")
 * - Connection arrows: ->
 * 
 * Example text input:
 *   [IGA:40A:2P] -> [SPD]
 *   [RCD:40A:30mA:2P] -> [PIA:16A:2.5mm:T20] -> C2
 */

export function parseUnifilarText(rawText) {
  if (!rawText) return [];
  
  const lines = rawText.split('\n');
  const parsedNodes = [];

  lines.forEach((line, index) => {
    const cleanLine = line.trim();
    // Skip empty lines or comment lines
    if (!cleanLine || cleanLine.startsWith('#') || cleanLine.startsWith('//')) {
      return;
    }

    const segments = cleanLine.split('->').map(s => s.trim());
    const lineNodes = [];

    segments.forEach((seg, segIndex) => {
      const bracketMatch = seg.match(/^\[(.*)\]$/);
      
      if (bracketMatch) {
        const parts = bracketMatch[1].split(':').map(p => p.trim());
        const type = parts[0] || 'UNKNOWN';
        
        let rating = '';
        let poles = '';
        let cable = '';
        let tube = '';
        
        // Parse parameters dynamically using regex checks
        for (let i = 1; i < parts.length; i++) {
          const part = parts[i];
          if (/^\d+A$/.test(part)) {
            rating = part;
          } else if (/^\d+P$/.test(part)) {
            poles = part;
          } else if (/^\d+mA$/.test(part)) {
            rating = rating ? `${rating} / ${part}` : part;
          } else if (/^[\d\.]+(?:mm²?)$/.test(part)) {
            cable = part;
          } else if (/^(?:T\d+|\d+mm)$/i.test(part)) {
            tube = part;
          } else {
            // Default fallback
            if (!rating) rating = part;
          }
        }

        lineNodes.push({
          id: `node-${index}-${segIndex}`,
          lineIndex: index,
          type,
          rating,
          poles,
          cable,
          tube,
          raw: seg
        });
      } else {
        // Plain text target circuit node
        lineNodes.push({
          id: `node-${index}-${segIndex}`,
          lineIndex: index,
          type: 'CIRCUIT',
          name: seg,
          raw: seg
        });
      }
    });

    // Link nodes on the same line to form connections
    for (let i = 0; i < lineNodes.length; i++) {
      const currentNode = lineNodes[i];
      const nextNode = lineNodes[i + 1];
      if (nextNode) {
        currentNode.target = nextNode.type === 'CIRCUIT' ? nextNode.name : nextNode.type;
        currentNode.targetId = nextNode.id;
      }
      parsedNodes.push(currentNode);
    }
  });

  return parsedNodes;
}
