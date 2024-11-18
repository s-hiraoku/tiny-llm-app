"use client";

import React, { useState } from "react";
import * as ort from "onnxruntime-web";
import { AutoTokenizer } from "@xenova/transformers";

export default function OnnxQA() {
  const [question, setQuestion] = useState("");
  const [context, setContext] = useState("");
  const [answer, setAnswer] = useState("");

  const handleRunInference = async () => {
    try {
      const tokenizer = await AutoTokenizer.from_pretrained(
        "ybelkada/japanese-roberta-question-answering"
      );

      const inputs = await tokenizer(question, context, {
        padding: true,
        truncation: true,
        max_length: 512,
        return_tensors: "pt",
      });
      const inputIds = Array.from(inputs.input_ids.data);
      const attentionMask = Array.from(inputs.attention_mask.data);

      if (!inputIds || !attentionMask) {
        throw new Error("トークナイズ結果が正しく生成されませんでした");
      }

      const session = await ort.InferenceSession.create("/qa_model.onnx");

      const inputIdsBigInt = new BigInt64Array(inputIds.map((x) => BigInt(x)));
      const attentionMaskBigInt = new BigInt64Array(
        attentionMask.map((x) => BigInt(x))
      );

      const inputIdsTensor = new ort.Tensor("int64", inputIdsBigInt, [
        1,
        inputIds.length,
      ]);
      const attentionMaskTensor = new ort.Tensor("int64", attentionMaskBigInt, [
        1,
        attentionMask.length,
      ]);

      const outputs = await session.run({
        input_ids: inputIdsTensor,
        attention_mask: attentionMaskTensor,
      });

      const startLogits = outputs.start_logits.data;
      const endLogits = outputs.end_logits.data;

      console.log("Start Logits:", startLogits, "End Logits:", endLogits);

      const startIndex = startLogits.indexOf(Math.max(...startLogits));
      const endIndex = endLogits.indexOf(Math.max(...endLogits));

      console.log("Start Index:", startIndex, "End Index:", endIndex);

      if (
        startIndex < 0 ||
        endIndex < 0 ||
        startIndex >= inputIds.length ||
        endIndex >= inputIds.length
      ) {
        throw new Error("Invalid start or end index.");
      }

      const tokens = inputIds.slice(startIndex, endIndex + 1);
      console.log("Tokens to decode:", tokens);

      if (!tokens || tokens.length === 0) {
        throw new Error("Tokens array is empty.");
      }

      const decodedAnswer = await tokenizer.decode(tokens, {
        skip_special_tokens: true,
      });

      setAnswer(decodedAnswer);
    } catch (error) {
      console.error("推論エラー:", error);
      setAnswer("エラーが発生しました");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white shadow-lg rounded-lg p-8 w-full max-w-2xl">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">
          質問応答システム
        </h1>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            コンテキスト:
          </label>
          <textarea
            value={context}
            onChange={(e) => setContext(e.target.value)}
            className="w-full border border-gray-300 rounded-md px-4 py-2 text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            rows={5}
            placeholder="コンテキストを入力してください"
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            質問:
          </label>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="w-full border border-gray-300 rounded-md px-4 py-2 text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="質問を入力してください"
          />
        </div>

        <button
          onClick={handleRunInference}
          className="w-full bg-blue-500 text-white font-semibold py-2 px-4 rounded-md hover:bg-blue-600 transition"
        >
          実行
        </button>
        <div className="mt-6">
          <h2 className="text-lg font-bold text-gray-800 mb-2">回答:</h2>
          <p className="bg-gray-100 border border-gray-300 rounded-md p-4 text-gray-800">
            {answer || "回答がここに表示されます"}
          </p>
        </div>
      </div>
    </div>
  );
}
