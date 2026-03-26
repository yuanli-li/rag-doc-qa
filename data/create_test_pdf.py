from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch


def create_neuro_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)

    # --- Page 1: Abstract & Intro ---
    c.setFont('Helvetica-Bold', 18)
    c.drawCentredString(4.25 * inch, 10 * inch,
                        "Pathological Mechanisms of Alpha-Synuclein")

    c.setFont('Helvetica-Bold', 12)
    c.drawString(1 * inch, 9 * inch, "Abstract")
    c.setFont('Helvetica', 10)
    # 模拟双栏：左栏
    text_left = [
        "This study investigates the aggregation of alpha-synuclein",
        "in felines. We observed significant neural recovery.",
        "The presence of Beta-amyloid plaques was also noted."
    ]
    y = 8.7 * inch
    for line in text_left:
        c.drawString(1 * inch, y, line)
        y -= 0.2 * inch

    # 模拟双栏：右栏 (测试 pypdf 是否会横向误读)
    c.setFont('Helvetica-Bold', 12)
    c.drawString(4.5 * inch, 9 * inch, "Key Findings")
    c.setFont('Helvetica', 10)
    text_right = [
        "1. Sleep patterns directly affect Tau phosphorylation.",
        "2. High-protein diets (20% increase) improve memory.",
        "3. Incubation at 37 degrees Celsius is optimal."
    ]
    y = 8.7 * inch
    for line in text_right:
        c.drawString(4.5 * inch, y, line)
        y -= 0.2 * inch

    c.showPage()

    # --- Page 2: Methods (双栏详细内容) ---
    c.setFont('Helvetica-Bold', 14)
    c.drawString(1 * inch, 10 * inch, "Materials and Methods")
    c.setFont('Helvetica', 10)
    c.drawString(1 * inch, 9.5 * inch,
                 "Samples were treated with 50 micrograms/mL of PFFs.")
    c.drawString(1 * inch, 9.3 * inch,
                 "Incubation period: 120 hours in a controlled environment.")
    c.save()
    print(f"Successfully created {filename}")


if __name__ == "__main__":
    create_neuro_pdf('neuro_paper.pdf')
