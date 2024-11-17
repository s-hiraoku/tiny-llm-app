import { NextRequest, NextResponse } from "next/server";
import { AutoTokenizer } from "@xenova/transformers";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { question, context } = body;

    if (!question || !context) {
      return NextResponse.json(
        { error: "Missing question or context" },
        { status: 400 }
      );
    }

    // トークナイザーをロード
    const tokenizer = await AutoTokenizer.from_pretrained(
      "distilbert-base-uncased-distilled-squad"
    );

    // トークナイズ処理
    const inputs = await tokenizer(question, context, {
      padding: true,
      truncation: true,
      max_length: 512,
      return_tensors: "np",
    });

    // トークナイズ済みデータを返却
    return NextResponse.json({
      input_ids: inputs.input_ids.data,
      attention_mask: inputs.attention_mask.data,
    });
  } catch (error) {
    console.error("トークナイザーエラー:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

export const runtime = "nodejs";
