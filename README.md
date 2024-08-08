# Veterinary LLM

## Video Demo
[Clike Here](https://youtu.be/63Jtx7NCO-Q)
## Project Overview
The Veterinary Large Language Model (LLM) aims to leverage advancements in artificial intelligence and natural language processing to develop a specialized model for veterinary medicine. This project focuses on creating a model capable of understanding and generating human-like text in the context of veterinary diagnostics, treatment, and general animal healthcare. The ultimate goal is to enhance the accessibility and quality of veterinary care through intelligent, context-aware, and responsive AI-driven solutions.

## System Architecture
<img src="https://github.com/ShuaimingJing/AIPI561/blob/main/assets/architecture.png" width="1024"/>


## Usage Examples & Screenshots

<img src="https://github.com/ShuaimingJing/AIPI561/blob/main/assets/use1.png" width="1024"/>
<img src="https://github.com/ShuaimingJing/AIPI561/blob/main/assets/use2.png" width="1024"/>

## How to Set Up
1. Download [mistral-7b-instruct-v0.2.Q4_0.llamafile](https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q4_0.llamafile?download=true)(3.85GB)
2. Open your computer's terminal
3. If you're using macOS, Linux, or BSD, you'll need to grant permission
   for your computer to execute this new file. (You only need to do this
   once.)

```sh
chmod +x mistral-7b-instruct-v0.2.Q4_0.llamafile
```

4. If you're on Windows, rename the file by adding ".exe" on the end.

5. Run the llamafile. e.g.:

```sh
./mistral-7b-instruct-v0.2.Q4_0.llamafile
```

6. Your browser should open automatically and display a chat interface.
   (If it doesn't, just open your browser and point it at http://localhost:8080)

7. When you're done chatting, return to your terminal and hit
   `Control-C` to shut down llamafile.

## Performance & Evaluation
The evaluation of Veterinary LLM consists of two parts: **Latency** and **Accuracy**. 10 questions are selected to test the performance. Latency measures the time taken by the LLM to generate responses. Then human-evaluation is used to measure the accuracy of the generated responses. The selected questions are being asked to Veterinary LLM, based model (Mistral-7B), and GPT-4o. Accuray has 10 ranks: 1-10, 10 being the highest.

A. What is the most common Basilus species found in dogs?

B. How does the dog affect the dog with a potato and pulse ingredients?

C. What can Coccidioides can cause dogs to dogs?

D. What is the average life expectancy of dogs with heart disease?

E. If the cat's total thyroid hormone level is low, what clinical symptoms can occur?

F. What was the diagnosis of the newborn fats of the horse?

G. Is the heart murmurs commonly found in a healthy cat?

H. What is the most common cause of the digestive tract foreign substance (FB) in livestock pigs?

I. What clinical discovery did cows with skin diseases?

J. Why is the width of red blood cells and platelet ratios in the cubs?

**Results for Latency Test for Veterinary LLM** 
|Questions|Latency (ms)|
|---------|--------|
|A |18733|
|B |17395|
|C |17443|
|D |18671|
|E |17174|
|F |19397|
|G |17459|
|H |18810|
|I |17124|
|J |16312|

**Results for Accuracy Test** 
|Questions|Veterinary LLM|Base Model (Mistral-7B)| GPT 4o|
|---------|--------|---------|---|
|A |8|4|7|
|B |7|7|9|
|C |7|6|8|
|D |8|6|8|
|E |6|5|9|
|F |8|4|6|
|G |9|6|9|
|H |7|7|9|
|I |8|5|5|
|J |5|7|7|
|average|7.3|5.7|7.5

**Performance Conclusion:** 
The average latency is about 18s for each question, which is higher than I expected. This is probably because the model is ran locally rather than on cloud. According to the table for accuracy test, the base model has the worst performance. GPT 4o has slightly higher performance than Veterinary LLM. Since the RAG pipeline is built on the base model, as we can see the performance improved a lot using RAG.

