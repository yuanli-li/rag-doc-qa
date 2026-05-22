from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch


def create_protocol_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)

    # --- 第 1 页：试剂与准备 ---
    c.setFont('Helvetica-Bold', 16)
    c.drawString(1 * inch, 10.5 * inch,
                 "Protocol: In Vitro Tau Aggregation Assay")

    c.setFont('Helvetica-Bold', 12)
    c.drawString(1 * inch, 9.8 * inch, "1. Reagents and Buffer Preparation")

    c.setFont('Helvetica', 10)
    lines_p1 = [
        "- Recombinant Tau (1-441): Purified according to lab standard SOP-03.",
        "- Heparin (H3393): Dissolved in water at a stock concentration of 10 mg/mL.",
        "- Thioflavin T (ThT): Prepared as a 1 mM stock solution, filtered at 0.22 um.",
        "- Buffer: 10 mM HEPES, 100 mM NaCl, pH 7.4.",
        "Crucial Note: All reagents must be kept on ice during the assembly."
    ]
    y = 9.5 * inch
    for line in lines_p1:
        c.drawString(1 * inch, y, line)
        y -= 0.25 * inch

    c.setFont('Helvetica-Oblique', 9)
    c.drawString(1 * inch, 1 * inch,
                 "Page 1 - Proprietary Content of Chao Peng Lab")
    c.showPage()

    # --- 第 2 页：操作步骤 ---
    c.setFont('Helvetica-Bold', 12)
    c.drawString(1 * inch, 10 * inch, "2. Assay Assembly")

    c.setFont('Helvetica', 10)
    lines_p2 = [
        "Step 1: Dilute Tau protein to a final concentration of 10 micro-molar (uM) in buffer.",
        "Step 2: Add Heparin to the mixture at a 1:1 weight ratio relative to Tau.",
        "Step 3: Add ThT to a final concentration of 20 uM.",
        "Step 4: Pipette 100 uL of the mixture into a black 96-well plate (half-area).",
        "Step 5: Seal the plate and incubate at 37 degrees Celsius with continuous shaking (300 rpm).",
        "Measurement: Record ThT fluorescence every 10 minutes (Ex: 440 nm, Em: 480 nm)."
    ]
    y = 9.5 * inch
    for line in lines_p2:
        c.drawString(1 * inch, y, line)
        y -= 0.25 * inch

    c.setFont('Helvetica-Oblique', 9)
    c.drawString(1 * inch, 1 * inch,
                 "Page 2 - Confidentially handled by Shujing")
    c.save()
    print(f"Successfully created {filename}")


if __name__ == "__main__":
    create_protocol_pdf('tau_protocol.pdf')
