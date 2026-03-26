import argparse
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import traceback
import unicodedata
import uuid

from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
)
from ai_scientist.utils.model_config import create_client
from ai_scientist.utils.prompt_manager import get_prompt

from ai_scientist.tools.open_alex import search_for_papers as open_alex_search
from ai_scientist.tools.semantic_scholar import search_for_papers as semantic_scholar_search
from ai_scientist.tools.pubmed import search_for_papers as pubmed_search
from ai_scientist.tools.open_alex import get_default_search_tool_name

# Determine default search tool from config.yaml
_default_tool_name = get_default_search_tool_name()

# Search tool registry: name -> (search_func, display_name)
_search_tools = {
    "SearchOpenAlex": (open_alex_search, "OpenAlex"),
    "SearchSemanticScholar": (semantic_scholar_search, "Semantic Scholar"),
    "SearchPubMed": (pubmed_search, "PubMed"),
}

def search_for_papers(query, result_limit=10):
    """Search for papers using configured default tool, with fallback.

    Uses the tool specified in config.yaml academic_search.default_tool.
    If the default tool fails, falls back to the other tools.
    """
    # Build ordered list: default tool first, then the rest as fallbacks
    default_func, default_name = _search_tools.get(
        _default_tool_name, (open_alex_search, "OpenAlex")
    )
    fallbacks = [
        (func, name) for key, (func, name) in _search_tools.items()
        if key != _default_tool_name
    ]

    # Try default tool first
    try:
        papers = default_func(query, result_limit)
        if papers:
            return papers
    except Exception as e:
        print(f"{default_name} search failed: {e}")

    # Try fallbacks
    for func, name in fallbacks:
        try:
            papers = func(query, result_limit)
            if papers:
                return papers
        except Exception as e:
            print(f"{name} search failed: {e}")

    return None

from ai_scientist.perform_vlm_review import generate_vlm_img_review
from ai_scientist.vlm import create_client as create_vlm_client
from ai_scientist.utils.model_config import load_model_config


def remove_accents_and_clean(s):
    # print("Original:", s)
    # Normalize to separate accents
    nfkd_form = unicodedata.normalize("NFKD", s)
    # Remove non-ASCII characters
    ascii_str = nfkd_form.encode("ASCII", "ignore").decode("ascii")
    # Remove anything but letters, digits, underscores, colons, dashes, @, {, }, and now commas
    ascii_str = re.sub(r"[^a-zA-Z0-9:_@\{\},-]+", "", ascii_str)
    # Convert to lowercase
    ascii_str = ascii_str.lower()
    # print("Cleaned: ", ascii_str)
    return ascii_str


def compile_latex(cwd, pdf_file, timeout=30):
    print("GENERATING LATEX")

    # 使用 XeLaTeX 编译以支持中文 (ctex 包需要 XeLaTeX)
    commands = [
        ["xelatex", "-interaction=nonstopmode", "template.tex"],
        ["bibtex", "template"],
        ["xelatex", "-interaction=nonstopmode", "template.tex"],
        ["xelatex", "-interaction=nonstopmode", "template.tex"],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)
        except subprocess.TimeoutExpired:
            print(
                f"EXCEPTION in compile_latex: LaTeX timed out after {timeout} seconds."
            )
            print(traceback.format_exc())
        except subprocess.CalledProcessError:
            print(
                f"EXCEPTION in compile_latex: Error running command {' '.join(command)}"
            )
            print(traceback.format_exc())

    print("FINISHED GENERATING LATEX")

    try:
        shutil.move(osp.join(cwd, "template.pdf"), pdf_file)
    except FileNotFoundError:
        print("Failed to rename PDF.")
        print("EXCEPTION in compile_latex while moving PDF:")
        print(traceback.format_exc())


def detect_pages_before_impact(latex_folder, timeout=30):
    """
    Temporarily copy the latex folder, compile, and detect on which page
    the phrase "Impact Statement" appears.
    Returns a tuple (page_number, line_number) if found, otherwise None.
    """
    temp_dir = osp.join(latex_folder, f"_temp_compile_{uuid.uuid4().hex}")
    try:
        shutil.copytree(latex_folder, temp_dir, dirs_exist_ok=True)

        # Compile in the temp folder
        # 使用 XeLaTeX 编译以支持中文 (ctex 包需要 XeLaTeX)
        commands = [
            ["xelatex", "-interaction=nonstopmode", "template.tex"],
            ["bibtex", "template"],
            ["xelatex", "-interaction=nonstopmode", "template.tex"],
            ["xelatex", "-interaction=nonstopmode", "template.tex"],
        ]
        for command in commands:
            try:
                subprocess.run(
                    command,
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return None

        temp_pdf_file = osp.join(temp_dir, "template.pdf")
        if not osp.exists(temp_pdf_file):
            return None

        # Try page-by-page extraction to detect "Impact Statement"
        for i in range(1, 51):
            page_txt = osp.join(temp_dir, f"page_{i}.txt")
            subprocess.run(
                [
                    "pdftotext",
                    "-f",
                    str(i),
                    "-l",
                    str(i),
                    "-q",
                    temp_pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                break
            with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                page_content = fp.read()
            lines = page_content.split("\n")
            for idx, line in enumerate(lines):
                if "Impact Statement" in line:
                    return (i, idx + 1)
        return None
    except Exception:
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_citation_addition(
    client, model, context, current_round, total_rounds, idea_text
):
    report, citations = context
    msg_history = []
    # 从 prompt.yaml 获取引用收集系统提示词
    citation_system_msg_template = get_prompt('writeup.citation_system')

    # 从 prompt.yaml 获取引用收集第一轮提示词
    citation_first_prompt_template = get_prompt('writeup.citation_first')

    # 从 prompt.yaml 获取引用收集第二轮提示词
    citation_second_prompt_template = get_prompt('writeup.citation_second')

    try:
        text, msg_history = get_response_from_llm(
            msg=citation_first_prompt_template.format(
                current_round=current_round + 1,
                total_rounds=total_rounds,
                Idea=idea_text,
                report=report,
                citations=citations,
            ),
            client=client,
            model=model,
            system_message=citation_system_msg_template.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "No more citations needed" in text:
            print("No more citations needed.")
            return None, True

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        query = json_output["Query"]
        papers = search_for_papers(query)
    except Exception:
        print("EXCEPTION in get_citation_addition (initial search):")
        print(traceback.format_exc())
        return None, False

    if papers is None:
        print("No papers found.")
        return None, False

    paper_strings = []
    for i, paper in enumerate(papers):
        paper_strings.append(
            "{i}: {title}. {authors}. {venue}, {year}.\nAbstract: {abstract}".format(
                i=i,
                title=paper["title"],
                authors=paper["authors"],
                venue=paper["venue"],
                year=paper["year"],
                abstract=paper["abstract"],
            )
        )
    papers_str = "\n\n".join(paper_strings)

    try:
        text, msg_history = get_response_from_llm(
            msg=citation_second_prompt_template.format(
                papers=papers_str,
                current_round=current_round + 1,
                total_rounds=total_rounds,
            ),
            client=client,
            model=model,
            system_message=citation_system_msg_template.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "Do not add any" in text:
            print("Do not add any.")
            return None, False

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        desc = json_output["Description"]
        selected_papers = str(json_output["Selected"])

        if selected_papers != "[]":
            selected_indices = []
            for x in selected_papers.strip("[]").split(","):
                x_str = x.strip().strip('"').strip("'")
                if x_str:
                    selected_indices.append(int(x_str))
            assert all(
                [0 <= i < len(papers) for i in selected_indices]
            ), "Invalid paper index"
            bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]

            cleaned_bibtexs = []
            for bibtex in bibtexs:
                newline_index = bibtex.find("\n")
                cite_key_line = bibtex[:newline_index]
                cite_key_line = remove_accents_and_clean(cite_key_line)
                cleaned_bibtexs.append(cite_key_line + bibtex[newline_index:])
            bibtexs = cleaned_bibtexs

            bibtex_string = "\n".join(bibtexs)
        else:
            return None, False

    except Exception:
        print("EXCEPTION in get_citation_addition (selecting papers):")
        print(traceback.format_exc())
        return None, False

    references_format = """% {description}
{bibtex}"""

    references_prompt = references_format.format(bibtex=bibtex_string, description=desc)
    return references_prompt, False


# 从 prompt.yaml 获取撰写系统提示词模板（使用 page_limit 参数格式化）
def get_writeup_system_template(page_limit: int) -> str:
    """获取撰写系统提示词模板"""
    system_template = get_prompt('writeup.system_template')
    data_usage_note = get_prompt('writeup.data_usage_note')
    conclusion_guidance = get_prompt('writeup.conclusion_guidance')

    # 组合所有提示词
    full_system_message = f"""{system_template}

{data_usage_note}

{conclusion_guidance}
"""
    return full_system_message.format(page_limit=page_limit)


# 从 prompt.yaml 获取撰写提示词
writeup_prompt = get_prompt('writeup.writeup_prompt')


def perform_writeup(
    base_folder,
    no_writing=False,
    num_cite_rounds=20,
    small_model="small_model",
    big_model="writeup",
    n_writeup_reflections=3,
    page_limit=8,
):
    compile_attempt = 0
    base_pdf_file = osp.join(base_folder, f"{osp.basename(base_folder)}")
    latex_folder = osp.join(base_folder, "latex")

    # Cleanup any previous latex folder and pdf
    if osp.exists(latex_folder):
        shutil.rmtree(latex_folder)
    # if osp.exists(pdf_file):
    #     os.remove(pdf_file)

    try:
        # Load idea text
        idea_text = ""
        research_idea_path = osp.join(base_folder, "research_idea.md")
        if osp.exists(research_idea_path):
            with open(research_idea_path, "r") as f_idea:
                idea_text = f_idea.read()
        else:
            idea_md_path = osp.join(base_folder, "idea.md")
            if osp.exists(idea_md_path):
                with open(idea_md_path, "r") as f_idea:
                    idea_text = f_idea.read()

        # Load summaries
        summary_files = [
            ("logs/0-run/baseline_summary.json", "BASELINE_SUMMARY"),
            ("logs/0-run/research_summary.json", "RESEARCH_SUMMARY"),
            ("logs/0-run/ablation_summary.json", "ABLATION_SUMMARY"),
        ]
        loaded_summaries = {}
        for fname, key in summary_files:
            path = osp.join(base_folder, fname)
            if osp.exists(path):
                try:
                    with open(path, "r") as f:
                        loaded_summaries[key] = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: {fname} is not valid JSON. Using empty data for {key}."
                    )
                    loaded_summaries[key] = {}
            else:
                loaded_summaries[key] = {}

        # Convert them to one big JSON string for context
        combined_summaries_str = json.dumps(loaded_summaries, indent=2)

        # Prepare a new fresh latex folder
        if not osp.exists(osp.join(latex_folder, "template.tex")):
            shutil.copytree(
                "ai_scientist/blank_icml_latex", latex_folder, dirs_exist_ok=True
            )

        writeup_file = osp.join(latex_folder, "template.tex")
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        # Gather plot filenames from figures/ folder
        figures_dir = osp.join(base_folder, "figures")
        plot_names = []
        if osp.exists(figures_dir):
            for fplot in os.listdir(figures_dir):
                if fplot.lower().endswith(".png"):
                    plot_names.append(fplot)

        # Load aggregator script to include in the prompt
        aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
        aggregator_code = ""
        if osp.exists(aggregator_path):
            with open(aggregator_path, "r") as fa:
                aggregator_code = fa.read()
        else:
            aggregator_code = "No aggregator script found."

        if no_writing:
            compile_latex(latex_folder, base_pdf_file + ".pdf")
            return osp.exists(base_pdf_file + ".pdf")

        # Run small model for citation additions
        client, client_model = create_client(small_model)
        for round_idx in range(num_cite_rounds):
            with open(writeup_file, "r") as f:
                writeup_text = f.read()
            try:
                references_bib = re.search(
                    r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}",
                    writeup_text,
                    re.DOTALL,
                )
                if references_bib is None:
                    raise ValueError("No references.bib found in template.tex")
                citations_text = references_bib.group(1)
                context_for_citation = (combined_summaries_str, citations_text)

                addition, done = get_citation_addition(
                    client,
                    client_model,
                    context_for_citation,
                    round_idx,
                    num_cite_rounds,
                    idea_text,
                )
                if done:
                    break

                if addition is not None:
                    # Simple check to avoid duplicating the same title
                    title_match = re.search(r" title = {(.*?)}", addition)
                    if title_match:
                        new_title = title_match.group(1).lower()
                        existing_titles = re.findall(
                            r" title = {(.*?)}", citations_text
                        )
                        existing_titles = [t.lower() for t in existing_titles]
                        if new_title not in existing_titles:
                            pattern_end = r"\end{filecontents}"
                            revised = writeup_text.replace(
                                pattern_end, f"\n{addition}{pattern_end}"
                            )
                            with open(writeup_file, "w") as fo:
                                fo.write(revised)
            except Exception:
                print("EXCEPTION in perform_writeup (citation round):")
                print(traceback.format_exc())
                continue

        # Generate VLM-based descriptions but do not overwrite plot_names
        try:
            # Use the vlm model type from config for VLM tasks
            vlm_client, vlm_model = create_vlm_client("vlm")
            desc_map = {}
            for pf in plot_names:
                ppath = osp.join(figures_dir, pf)
                if not osp.exists(ppath):
                    continue
                img_dict = {
                    "images": [ppath],
                    "caption": "No direct caption",
                }
                review_data = generate_vlm_img_review(img_dict, vlm_model, vlm_client)
                if review_data:
                    desc_map[pf] = review_data.get(
                        "Img_description", "No description found"
                    )
                else:
                    desc_map[pf] = "No description found"

            # Prepare a string listing all figure descriptions in order
            plot_descriptions_list = []
            for fname in plot_names:
                desc_text = desc_map.get(fname, "No description found")
                plot_descriptions_list.append(f"{fname}: {desc_text}")
            plot_descriptions_str = "\n".join(plot_descriptions_list)
        except Exception:
            print("EXCEPTION in VLM figure description generation:")
            print(traceback.format_exc())
            plot_descriptions_str = "No descriptions available."

        # Construct final prompt for big model, placing the figure descriptions alongside the plot list
        # 从 prompt.yaml 获取撰写系统提示词模板
        big_model_system_message = get_writeup_system_template(page_limit=page_limit)
        big_client, big_client_model = create_client(big_model)
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        combined_prompt = writeup_prompt.format(
            idea_text=idea_text,
            summaries=combined_summaries_str,
            aggregator_code=aggregator_code,
            plot_list=", ".join(plot_names),
            latex_writeup=writeup_text,
            plot_descriptions=plot_descriptions_str,
        )

        response, msg_history = get_response_from_llm(
            msg=combined_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            print_debug=False,
        )

        latex_code_match = re.search(r"```latex(.*?)```", response, re.DOTALL)
        if not latex_code_match:
            return False
        updated_latex_code = latex_code_match.group(1).strip()
        with open(writeup_file, "w") as f:
            f.write(updated_latex_code)

        # Multiple reflection loops on the final LaTeX
        for i in range(n_writeup_reflections):
            with open(writeup_file, "r") as f:
                current_latex = f.read()

            # Check for unused or invalid figure references
            referenced_figs_temp = re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}", current_latex
            )
            used_figs = set(os.path.basename(fig) for fig in referenced_figs_temp)
            all_figs = set(plot_names)
            unused_figs = all_figs - used_figs
            invalid_figs = used_figs - all_figs

            # Compile current version before reflection
            compile_latex(latex_folder, base_pdf_file + f"_{compile_attempt}.pdf")
            compile_attempt += 1
            print(f"Compiled {base_pdf_file}_{compile_attempt}.pdf")

            # Detect where "Impact Statement" appears
            impact_loc = detect_pages_before_impact(latex_folder)
            if impact_loc is not None:
                page_num, line_num = impact_loc
                reflection_page_info = (
                    f"\nCurrently, 'Impact Statement' begins on page {page_num}, approximately on line {line_num}. "
                    f"The page limit is {page_limit}, which is before the Impact Statement. "
                    f"Papers often look more professional if the main text is near or just under {page_limit} pages in length.\n"
                )
            else:
                reflection_page_info = "\nCould not detect 'Impact Statement' page (compilation or detection failed).\n"

            check_output = os.popen(
                f"chktex {writeup_file} -q -n2 -n24 -n13 -n1"
            ).read()

            # 从 prompt.yaml 获取反思提示词
            reflection_prompt = get_prompt(
                'writeup.reflection',
                unused_figs=str(sorted(unused_figs)),
                invalid_figs=str(sorted(invalid_figs)),
                reflection_page_info=reflection_page_info,
                check_output=check_output
            )

            reflection_response, msg_history = get_response_from_llm(
                msg=reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history,
                print_debug=False,
            )

            if "I am done" in reflection_response:
                print(
                    "LLM indicated it is done with reflections. Exiting reflection loop."
                )
                break

            reflection_code_match = re.search(
                r"```latex(.*?)```", reflection_response, re.DOTALL
            )
            if reflection_code_match:
                reflected_latex_code = reflection_code_match.group(1).strip()
                if reflected_latex_code != current_latex:
                    final_text = reflected_latex_code
                    cleanup_map = {
                        "</end": r"\\end",
                        "</begin": r"\\begin",
                        "’": "'",
                    }
                    for bad_str, repl_str in cleanup_map.items():
                        final_text = final_text.replace(bad_str, repl_str)
                    final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)

                    compile_latex(
                        latex_folder, base_pdf_file + f"_{compile_attempt}.pdf"
                    )
                    compile_attempt += 1
                    print(f"Compiled {base_pdf_file}_{compile_attempt}.pdf")
                else:
                    print(f"No changes in reflection step {i+1}.")
                    break
            else:
                print(f"No valid LaTeX code block found in reflection step {i+1}.")
                break

        return osp.exists(base_pdf_file + f"_{compile_attempt-1}.pdf")

    except Exception:
        print("EXCEPTION in perform_writeup:")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform writeup for a project")
    parser.add_argument("--folder", type=str, help="Project folder", required=True)
    parser.add_argument("--no-writing", action="store_true", help="Only generate")
    parser.add_argument("--num-cite-rounds", type=int, default=20)
    parser.add_argument(
        "--model",
        type=str,
        default="small_model",
        help="Model type to use for citation collection (references config.yaml).",
    )
    parser.add_argument(
        "--big-model",
        type=str,
        default="writeup",
        help="Model type to use for final writeup (references config.yaml).",
    )
    parser.add_argument(
        "--writeup-reflections",
        type=int,
        default=3,
        help="Number of reflection steps for the final LaTeX writeup.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=8,
        help="Target page limit for the main paper (excluding references, impact statement, etc.)",
    )
    args = parser.parse_args()

    try:
        success = perform_writeup(
            base_folder=args.folder,
            no_writing=args.no_writing,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model,
            big_model=args.big_model,
            n_writeup_reflections=args.writeup_reflections,
            page_limit=args.page_limit,
        )
        if not success:
            print("Writeup process did not complete successfully.")
    except Exception:
        print("EXCEPTION in main:")
        print(traceback.format_exc())
