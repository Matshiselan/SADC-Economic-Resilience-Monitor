import ollama
import re

REPORT_PATH = 'reports/sadc_ministerial_brief.txt'

PROMPT_TEMPLATE = '''
You are an expert macro-fiscal policy analyst. Your task is to generate a concise, policy-focused executive summary for the SADC Ministerial Brief, based on the following appendix for {country_code}.

Appendix for {country_code}:
{appendix_text}

Instructions:
- Summarize key macro-fiscal risks, trends, and policy recommendations.
- Highlight data quality issues and structural breaks if relevant.
- Use clear, actionable language suitable for Ministers of Finance and Central Bank Governors.
- Keep the summary under 250 words.
- Do not repeat table content; synthesize insights.

Executive Summary:
'''

def extract_appendices(report_path):
    with open(report_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split by appendix section
    sections = re.findall(r'Appendix: (\w+)\n(.*?)(?=\n=+|$)', text, re.DOTALL)
    return {country: appendix.strip() for country, appendix in sections}

def generate_summary(country_code, appendix_text):
    prompt = PROMPT_TEMPLATE.format(country_code=country_code, appendix_text=appendix_text)
    response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

if __name__ == '__main__':
    appendices = extract_appendices(REPORT_PATH)
    for country, appendix in appendices.items():
        summary = generate_summary(country, appendix)
        print(f"Ollama Executive Summary: {country}\n{summary}\n")
