"use client";

import { useState } from "react";
import { performQAInference } from "./qa-inference";

export default function QuestionAnswering() {
  const [question, setQuestion] = useState("");
  const [context, setContext] = useState("");
  const [result, setResult] = useState<{
    answer: string;
    score: number;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const inferenceResult = await performQAInference({
        question,
        context,
      });
      setResult(inferenceResult);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "推論中にエラーが発生しました"
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white shadow-lg rounded-lg p-8 w-full max-w-3xl">
        <h1 className="text-2xl font-bold mb-4">
          コンテキストの内容を答えるよシステム
        </h1>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="context" className="block text-sm font-medium mb-1">
              コンテキスト
            </label>
            <textarea
              id="context"
              value={context}
              onChange={(e) => setContext(e.target.value)}
              className="w-full p-2 border rounded-md"
              rows={4}
              required
            />
          </div>

          <div>
            <label
              htmlFor="question"
              className="block text-sm font-medium mb-1"
            >
              質問
            </label>
            <input
              type="text"
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="w-full p-2 border rounded-md"
              required
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600 disabled:opacity-50"
          >
            {isLoading ? "推論中..." : "回答を生成"}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-md">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-4 p-4 bg-gray-50 rounded-md">
            <h2 className="font-semibold mb-2">回答:</h2>
            <p className="mb-2">{result.answer}</p>
            <p className="text-sm text-gray-600">
              確信度スコア: {result.score.toFixed(2)}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
