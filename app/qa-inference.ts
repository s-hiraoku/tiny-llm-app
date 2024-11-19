import * as ort from "onnxruntime-web";
import { AutoTokenizer } from "@xenova/transformers";

interface InferenceInput {
  question: string;
  context: string;
}

interface InferenceResult {
  answer: string;
  score: number;
}

/**
 * 質問応答モデルによる推論を実行する関数
 */
export async function performQAInference({
  question,
  context,
  modelPath = "/qa-model.onnx",
  modelName = "ybelkada/japanese-roberta-question-answering",
  maxLength = 512,
}: InferenceInput & {
  modelPath?: string;
  modelName?: string;
  maxLength?: number;
}): Promise<InferenceResult> {
  try {
    // トークナイザーの初期化
    const tokenizer = await AutoTokenizer.from_pretrained(modelName);

    // 入力テキストのトークン化
    const tokens = await tokenizer(question, context, {
      padding: true,
      truncation: true,
      maxLength: maxLength,
      returnTensors: true,
    });

    // トークンデータの取得
    const inputIds = Array.from(tokens.input_ids.data);
    const attentionMask = Array.from(tokens.attention_mask.data);

    // ONNXランタイムセッションの初期化
    const session = await ort.InferenceSession.create(modelPath);

    // ONNX Runtimeで使用する入力形式に変換
    const feeds = {
      input_ids: new ort.Tensor("int64", inputIds, [1, inputIds.length]),
      attention_mask: new ort.Tensor("int64", attentionMask, [
        1,
        attentionMask.length,
      ]),
    };

    // 推論の実行
    const output = await session.run(feeds);

    // 開始位置と終了位置のロジットを取得
    const startLogits = Object.values(output.start_logits.cpuData);
    const endLogits = Object.values(output.end_logits.cpuData);

    // 最も確率の高い開始位置と終了位置を取得
    const startPosition = startLogits.indexOf(Math.max(...startLogits));
    const endPosition = endLogits.indexOf(Math.max(...endLogits));

    // 有効な開始位置と終了位置の確認
    if (
      startPosition === -1 ||
      endPosition === -1 ||
      startPosition >= endPosition ||
      startPosition >= inputIds.length ||
      endPosition >= inputIds.length
    ) {
      return {
        answer: "回答を抽出できませんでした",
        score: 0,
      };
    }

    // トークンを元のテキストに変換して回答を抽出
    const answer_tokens = inputIds.slice(startPosition, endPosition + 1);

    // answer_tokensが空でないことを確認
    if (answer_tokens.length === 0) {
      return {
        answer: "回答を抽出できませんでした",
        score: 0,
      };
    }

    const answer = await tokenizer.decode(answer_tokens);

    // スコアの計算（開始位置と終了位置の確率の平均）
    const score = (startLogits[startPosition] + endLogits[endPosition]) / 2;

    // 空の回答をチェック
    if (!answer || answer.trim() === "") {
      return {
        answer: "回答を抽出できませんでした",
        score: 0,
      };
    }

    return {
      answer,
      score,
    };
  } catch (error) {
    console.error("推論中にエラーが発生しました:", error);
    throw error;
  }
}
