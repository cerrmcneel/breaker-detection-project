/**
 * Client-side SVG exporter to high-resolution PNG or WebP images.
 */
export function exportSvgToImage(svgElement, filename, format = 'png', scale = 2) {
  if (!svgElement) return;

  const serializer = new XMLSerializer();
  const svgString = serializer.serializeToString(svgElement);
  const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  const img = new Image();
  img.onload = () => {
    // Size the canvas from the SVG's intrinsic viewBox (not getBoundingClientRect),
    // so the export stays full-resolution regardless of any on-screen zoom/pan
    // transform applied by the viewer wrapper.
    const vb = svgElement.viewBox && svgElement.viewBox.baseVal;
    const rect = svgElement.getBoundingClientRect();
    const width = (vb && vb.width) || rect.width || 1200;
    const height = (vb && vb.height) || rect.height || 800;
    
    const canvas = document.createElement('canvas');
    canvas.width = width * scale;
    canvas.height = height * scale;

    const ctx = canvas.getContext('2d');
    
    // Set solid white background for professional print/document integration
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw image at scaled size
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Clean up memory
    URL.revokeObjectURL(url);

    // Generate and trigger download link
    const mimeType = format === 'webp' ? 'image/webp' : 'image/png';
    const imageUri = canvas.toDataURL(mimeType, 1.0);
    
    const link = document.createElement('a');
    link.download = `${filename}.${format}`;
    link.href = imageUri;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  img.src = url;
}
