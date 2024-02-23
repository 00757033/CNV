from fpdf import FPDF

def txt_to_pdf(input_file, output_file):
    """Convert a text file into a PDF file.

    Args:
        input_file: The input text file.
        output_file: The output PDF file.

    Returns:
        None
    """

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    with open(input_file, 'r') as f:
        for line in f:
            pdf.cell(200, 10, txt=line, ln=1, align='L')
    pdf.output(output_file)
if __name__ == '__main__':

    # Example usage
    input_txt_file = './record/PCV_receive.txt'
    output_pdf_file = './record/PCV_receive.pdf'
    txt_to_pdf(input_txt_file, output_pdf_file)
