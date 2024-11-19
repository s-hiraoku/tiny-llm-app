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

    // ソフトマックス関数を使用して確率を計算
    function softmax(logits: number[]) {
      const max = Math.max(...logits);
      const exp = logits.map((x) => Math.exp(x - max));
      const sum = exp.reduce((a, b) => a + b, 0);
      return exp.map((x) => x / sum);
    }

    const softmaxStartLogits = softmax(startLogits);
    const softmaxEndLogits = softmax(endLogits);

    let startPosition = softmaxStartLogits.indexOf(
      Math.max(...softmaxStartLogits)
    );
    let endPosition = softmaxEndLogits.indexOf(Math.max(...softmaxEndLogits));

    // 有効な開始位置と終了位置の確認
    if (
      startPosition === -1 ||
      endPosition === -1 ||
      startPosition >= inputIds.length ||
      endPosition >= inputIds.length ||
      startPosition >= endPosition
    ) {
      // より詳細な診断情報を出力
      console.log("Diagnostic Information:", {
        startLogits,
        endLogits,
        softmaxStartLogits,
        softmaxEndLogits,
        startPosition,
        endPosition,
        inputIdsLength: inputIds.length,
      });

      // トップ5の候補位置を検出
      const topStartPositions = softmaxStartLogits
        .map((prob, index) => ({ prob, index }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5);

      const topEndPositions = softmaxEndLogits
        .map((prob, index) => ({ prob, index }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5);

      console.log("Top Start Positions:", topStartPositions);
      console.log("Top End Positions:", topEndPositions);

      // より賢明な位置選択
      const validStartPositions = topStartPositions.filter(
        (pos) => pos.index < inputIds.length
      );
      const validEndPositions = topEndPositions.filter(
        (pos) =>
          pos.index < inputIds.length &&
          pos.index > validStartPositions[0]?.index
      );

      if (validStartPositions.length > 0 && validEndPositions.length > 0) {
        startPosition = validStartPositions[0].index;
        endPosition = validEndPositions[0].index;
      } else {
        return {
          answer: "回答を抽出できませんでした",
          score: 0,
        };
      }
    }

    // トークンを元のテキストに変換して回答を抽出
    const answer_tokens = inputIds.slice(startPosition, endPosition + 1);

    // answer_tokensが空でないことを確認
    if (answer_tokens.length === 0) {
      console.log("🚀 ~ answer_tokens:", answer_tokens);
      return {
        answer: "回答を抽出できませんでした",
        score: 0,
      };
    }

    const answer = await tokenizer.decode(answer_tokens);

    // const answer = await tokenizer.decode(answer_tokens, {
    //   skip_special_tokens: true,
    //   clean_up_tokenization_spaces: false,
    // });

    // const answer = await tokenizer.decode(answer_tokens, {
    //   skip_special_tokens: true,
    //   clean_up_tokenization_spaces: true,
    //   group_tokens: true,
    // });

    console.log("Input IDs:", inputIds);
    console.log("Start Logits:", startLogits);
    console.log("End Logits:", endLogits);
    console.log("Start Position:", startPosition);
    console.log("End Position:", endPosition);
    console.log("Answer Tokens:", answer_tokens);

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
