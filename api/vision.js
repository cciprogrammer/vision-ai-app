/**
 * Vercel Serverless Function - Vision AI API
 * 通过后端代理解决 CORS 问题，调用 Hugging Face Qwen2-VL 模型
 */

export default async function handler(req, res) {
  // 设置 CORS 头
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // 处理 OPTIONS 预检请求
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // 只允许 POST 请求
  if (req.method !== 'POST') {
    return res.status(405).json({ error: '只支持 POST 请求' });
  }

  try {
    const { image, question, mode } = req.body;

    // 验证输入
    if (!image) {
      return res.status(400).json({ error: '请上传图片' });
    }

    // 获取 Hugging Face Token
    const HF_TOKEN = process.env.HF_TOKEN;
    if (!HF_TOKEN) {
      return res.status(500).json({ 
        error: '服务器未配置 Hugging Face Token',
        hint: '请在 Vercel 项目设置中添加 HF_TOKEN 环境变量'
      });
    }

    // 根据模式构建 prompt
    let prompt = '';
    switch (mode) {
      case 'analyze':
        prompt = '请详细分析这张图片，包括：1. 图片主体是什么 2. 图片场景/背景 3. 图片风格/色调 4. 图片中的文字或数据（如有）5. 其他值得注意的细节。请用中文回答。';
        break;
      case 'describe':
        prompt = '请用一句话简洁描述这张图片的主要内容。请用中文回答。';
        break;
      case 'qa':
      default:
        prompt = question || '请描述这张图片的内容。请用中文回答。';
        break;
    }

    console.log('调用 Hugging Face API，模式:', mode);

    // 调用 Hugging Face Inference API
    const response = await fetch(
      'https://api-inference.huggingface.co.co/models/Qwen/Qwen2-VL-2B-Instruct',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${HF_TOKEN}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputs: {
            image: image,
            question: prompt
          }
        }),
      }
    );

    // 检查响应状态
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('Hugging Face API 错误:', response.status, errorData);
      
      // 如果模型正在加载，返回特殊状态码让前端知道要重试
      if (response.status === 503) {
        return res.status(503).json({
          error: '模型正在加载中，请稍后重试',
          retry: true,
          estimated_time: errorData.estimated_time || 30
        });
      }
      
      return res.status(response.status).json({
        error: `API 请求失败: ${response.status}`,
        details: errorData
      });
    }

    // 处理流式响应
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('text/event-stream')) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      return response.body.pipe(res);
    }

    // 处理普通响应
    const data = await response.json();
    console.log('API 响应成功');

    // 提取回答内容
    let answer = '';
    if (typeof data === 'string') {
      answer = data;
    } else if (data.generated_text) {
      answer = data.generated_text;
    } else if (data.output && data.output.text) {
      answer = data.output.text;
    } else if (Array.isArray(data) && data.length > 0) {
      answer = typeof data[0] === 'string' ? data[0] : JSON.stringify(data[0]);
    } else if (data.answer) {
      answer = data.answer;
    } else {
      answer = JSON.stringify(data);
    }

    return res.status(200).json({
      success: true,
      answer: answer,
      model: 'Qwen/Qwen2-VL-2B-Instruct'
    });

  } catch (error) {
    console.error('API 调用错误:', error);
    return res.status(500).json({
      error: '服务器内部错误',
      message: error.message
    });
  }
}
