//! A stream is limited by the gap between chunks, not by its total length.
//!
//! The mock server writes each chunk after a pause. `timeout` on the client config
//! is shorter than the whole stream and longer than every pause.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use liter_llm::{ChatCompletionRequest, ClientConfigBuilder, DefaultClient, LiterLlmError, LlmClient};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// What the mock server does after it reads the request.
#[derive(Clone, Copy)]
struct Script {
    /// Pause before the response headers.
    header_pause: Duration,
    /// Pause before each content chunk.
    chunk_pause: Duration,
    /// Number of content chunks.
    chunks: usize,
    /// When true the server sends the first chunk, then stays silent and never finishes.
    stall_after_first: bool,
}

fn sse(delta: serde_json::Value, finish: Option<&str>) -> Vec<u8> {
    let chunk = serde_json::json!({
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    });
    let data = format!("data: {chunk}\n\n");
    format!("{:x}\r\n{data}\r\n", data.len()).into_bytes()
}

async fn serve_one(mut sock: TcpStream, script: Script) {
    let mut buf = vec![0u8; 16384];
    let mut seen = Vec::new();
    // Read until the end of the request headers; the body is small and arrives with them.
    while !seen.windows(4).any(|w| w == b"\r\n\r\n") {
        match sock.read(&mut buf).await {
            Ok(0) | Err(_) => return,
            Ok(n) => seen.extend_from_slice(&buf[..n]),
        }
    }
    tokio::time::sleep(script.header_pause).await;
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
    if sock.write_all(head.as_bytes()).await.is_err() {
        return;
    }
    let _ = sock
        .write_all(&sse(serde_json::json!({"role": "assistant", "content": ""}), None))
        .await;
    if script.stall_after_first {
        tokio::time::sleep(Duration::from_secs(30)).await;
        return;
    }
    for i in 0..script.chunks {
        tokio::time::sleep(script.chunk_pause).await;
        if sock
            .write_all(&sse(serde_json::json!({"content": format!("w{i} ")}), None))
            .await
            .is_err()
        {
            return;
        }
    }
    let _ = sock.write_all(&sse(serde_json::json!({}), Some("stop"))).await;
    let done = "data: [DONE]\n\n";
    let _ = sock
        .write_all(format!("{:x}\r\n{done}\r\n0\r\n\r\n", done.len()).as_bytes())
        .await;
}

async fn start(script: Script) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((sock, _)) = listener.accept().await {
            tokio::spawn(serve_one(sock, script));
        }
    });
    format!("http://{addr}/v1")
}

fn client(base: String, timeout: Duration) -> DefaultClient {
    let config = ClientConfigBuilder::new("k")
        .base_url(base)
        .timeout(timeout)
        .max_retries(0)
        .build();
    DefaultClient::new(config, Some("openai")).unwrap()
}

fn request() -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .unwrap()
}

#[tokio::test]
async fn stream_outlives_the_timeout_while_chunks_keep_arriving() {
    // 6 chunks, 400 ms apart: about 2.4 s in total, against a 1 s timeout.
    let base = start(Script {
        header_pause: Duration::ZERO,
        chunk_pause: Duration::from_millis(400),
        chunks: 6,
        stall_after_first: false,
    })
    .await;
    let c = client(base, Duration::from_secs(1));
    let started = Instant::now();
    let mut stream = c.chat_stream(request()).await.expect("stream opens");
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        let chunk = item.expect("a healthy stream must not fail");
        if let Some(t) = chunk.choices.first().and_then(|ch| ch.delta.content.as_deref()) {
            text.push_str(t);
        }
    }
    assert_eq!(text, "w0 w1 w2 w3 w4 w5 ");
    assert!(started.elapsed() > Duration::from_secs(2), "the stream must outlast the timeout");
}

#[tokio::test]
async fn stream_that_goes_silent_fails_with_the_timeout_error() {
    let base = start(Script {
        header_pause: Duration::ZERO,
        chunk_pause: Duration::ZERO,
        chunks: 0,
        stall_after_first: true,
    })
    .await;
    let c = client(base, Duration::from_millis(500));
    let started = Instant::now();
    let mut stream = c.chat_stream(request()).await.expect("stream opens");
    let mut last_err = None;
    while let Some(item) = stream.next().await {
        if let Err(e) = item {
            last_err = Some(e);
            break;
        }
    }
    let err = last_err.expect("a silent stream must fail");
    assert!(matches!(err, LiterLlmError::Timeout), "got {err:?}");
    assert!(started.elapsed() < Duration::from_secs(10), "the stall must end at the idle limit");
}

#[tokio::test]
async fn stream_with_no_response_headers_fails_with_the_timeout_error() {
    let base = start(Script {
        header_pause: Duration::from_secs(5),
        chunk_pause: Duration::ZERO,
        chunks: 0,
        stall_after_first: false,
    })
    .await;
    let c = client(base, Duration::from_millis(500));
    let result = c.chat_stream(request()).await;
    assert!(matches!(result, Err(LiterLlmError::Timeout)), "got {:?}", result.err());
}
