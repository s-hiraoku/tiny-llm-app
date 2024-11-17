"use client";

import React, { useState } from "react";
import * as ort from "onnxruntime-web";

export default function OnnxInferencePage() {
  const [question, setQuestion] = useState("イベントはいつ開催されますか？");
  const [context, setContext] = useState(
    "イベントは2024年11月15日に開催されます。場所は東京です。"
  );
  const [answer, setAnswer] = useState("");

  const handleRunInference = async () => {
    try {
      // トークナイズAPIを呼び出してトークナイズ済みデータを取得
      const response = await fetch("/api/tokenize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question, context }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to tokenize input");
      }

      const { input_ids, attention_mask } = await response.json();

      // ONNXモデルをロード
      const session = await ort.InferenceSession.create("/qa_model.onnx");

      // モデル入力を準備
      const inputIdsTensor = new ort.Tensor("int64", input_ids, [
        1,
        input_ids.length,
      ]);
      const attentionMaskTensor = new ort.Tensor("int64", attention_mask, [
        1,
        attention_mask.length,
      ]);

      // 推論を実行
      const outputs = await session.run({
        input_ids: inputIdsTensor,
        attention_mask: attentionMaskTensor,
      });

      // 出力から回答範囲を取得
      const startLogits = Array.from(outputs.start_logits.data as Float32Array);
      const endLogits = Array.from(outputs.end_logits.data as Float32Array);
      const startIndex = startLogits.indexOf(Math.max(...startLogits));
      const endIndex = endLogits.indexOf(Math.max(...endLogits));

      // デコードされたトークンを取得
      const tokens = input_ids.slice(startIndex, endIndex + 1);
      const decodedAnswer = tokens
        .map((id: number) => String.fromCharCode(id))
        .join("");

      setAnswer(decodedAnswer);
    } catch (error) {
      console.error("ONNX推論エラー:", error);
      setAnswer("エラーが発生しました");
    }
  };

  return (
    <div>
      <h1>ONNXモデルで質問応答</h1>
      <div>
        <label>
          質問:
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
        </label>
      </div>
      <div>
        <label>
          コンテキスト:
          <textarea
            value={context}
            onChange={(e) => setContext(e.target.value)}
          />
        </label>
      </div>
      <button onClick={handleRunInference}>推論を実行</button>
      <div>
        <h2>答え:</h2>
        <p>{answer}</p>
      </div>
    </div>
  );
}
