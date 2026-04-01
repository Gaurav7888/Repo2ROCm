import csv
import re
import requests
from bs4 import BeautifulSoup


def scrape_neurips_papers(url: str, output_csv: str) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("Could not find the papers table on the page")

    rows = table.find_all("tr")[1:]  # skip header

    papers = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        paper_cell = cells[1]
        code_cell = cells[3]

        title_tag = paper_cell.find("a")
        title = title_tag.get_text(strip=True) if title_tag else paper_cell.get_text(strip=True)
        title = re.sub(r"\s+", " ", title)

        code_tag = code_cell.find("a", href=True)
        code_url = code_tag["href"] if code_tag else ""

        if title and code_url:
            papers.append({"paper_name": title, "code_url": code_url})

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["paper_name", "code_url"])
        writer.writeheader()
        writer.writerows(papers)

    print(f"Saved {len(papers)} papers to {output_csv}")


if __name__ == "__main__":
    scrape_neurips_papers(
        url="https://www.paperdigest.org/2025/11/neurips-2025-papers-with-code-data/",
        output_csv="neurips2025_papers.csv",
    )
