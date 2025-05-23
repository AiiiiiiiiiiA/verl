import argparse
import json
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Format a JSONL dataset for the search-augmented LLM agent.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file. Each line should be a JSON object with 'question' and 'answer' fields."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the formatted JSONL data."
    )
    parser.add_argument(
        "--prompt_instruction_template",
        type=str,
        default="You are a helpful assistant. Your task is to answer the given question. You can use the <search>tool_query</search> tool to find information. Reason step-by-step using <think>your thoughts</think>. When you have the final answer, present it using <answer>your_answer</answer>. Here is the question:",
        help="String template for the initial instruction given to the LLM."
    )
    args = parser.parse_args()

    logger.info(f"Starting data formatting process.")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")

    # --- File Handling ---
    if not os.path.exists(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        return

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return

    # --- Process Data ---
    processed_lines = 0
    error_lines = 0

    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile, \
             open(args.output_file, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping line {i+1} due to JSON decode error: {line.strip()}")
                    error_lines += 1
                    continue

                original_question = data.get("question")
                original_answer = data.get("answer")

                if original_question is None or original_answer is None:
                    logger.warning(f"Skipping line {i+1} due to missing 'question' or 'answer' field. Data: {data}")
                    error_lines += 1
                    continue
                
                # Construct the new prompt
                new_prompt = f"{args.prompt_instruction_template}\n\nQuestion: {original_question}\n\nAssistant:"

                output_data = {
                    "prompt": new_prompt,
                    "ground_truth_answer": original_answer,
                    "original_question": original_question 
                }

                outfile.write(json.dumps(output_data) + '\n')
                processed_lines += 1

                if (processed_lines + error_lines) % 1000 == 0: # Log progress every 1000 lines
                    logger.info(f"Processed {processed_lines + error_lines} lines ({processed_lines} successful, {error_lines} errors)...")

    except IOError as e:
        logger.error(f"IOError during file processing: {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}")
        return

    logger.info(f"Data formatting complete.")
    logger.info(f"Total lines processed successfully: {processed_lines}")
    logger.info(f"Total lines with errors/skipped: {error_lines}")
    logger.info(f"Formatted data saved to: {args.output_file}")

if __name__ == "__main__":
    main()
