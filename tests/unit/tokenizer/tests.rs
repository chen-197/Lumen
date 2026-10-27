use super::*;

fn tokenizer() -> LlamaTokenizer {
    LlamaTokenizer::from_inner(HFTokenizer::from_bytes(br#"{
            "version":"1.0","truncation":null,"padding":null,
            "added_tokens":[{"id":10,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,
            "decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]},
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"prefix":1,"<0xE7>":2,"<0x9D>":3,"<0xA1>":4,"<0x9C>":5,"<0xA0>":6,"<0xF0>":7,"<0x80>":8,"literal":9,"</s>":10},"unk_token":"[UNK]"}
        }"#).unwrap())
}

#[test]
fn incomplete_byte_run_preserves_prior_complete_characters_and_raw_ids() {
    let t = tokenizer();
    let ids = vec![1, 2, 3, 4, 2, 5];
    let raw = t.decode(&ids, true);
    assert!(raw.contains('\u{fffd}'));
    let complete = t.decode_complete_prefix(&ids, true);
    assert_eq!(complete.text, "prefix睡");
    assert_eq!(complete.consumed_tokens, 4);
    assert_eq!(complete.pending_token_ids, vec![2, 5]);
    assert_eq!(t.decode(&ids, true), raw);
    assert_eq!(ids, vec![1, 2, 3, 4, 2, 5]);
    assert_eq!(
        t.decode_complete_prefix(&[1, 2, 3, 4, 2, 5, 6], true).text,
        "prefix睡眠"
    );
}

#[test]
fn byte_prefix_stream_is_monotonic_and_complete() {
    let t = tokenizer();
    let ids = [1, 2, 3, 4, 2, 5, 6];
    let mut previous = String::new();
    for n in 0..=ids.len() {
        let prefix = t.decode_complete_prefix(&ids[..n], true);
        assert!(prefix.text.starts_with(&previous));
        assert!(!prefix.text.contains('\u{fffd}'));
        previous = prefix.text;
    }
    assert_eq!(previous, "prefix睡眠");
}

#[test]
fn invalid_bytes_are_visible_and_skipped_specials_do_not_flush_pending_bytes() {
    let t = tokenizer();
    let invalid = [1, 8, 2];
    let prefix = t.decode_complete_prefix(&invalid, true);
    assert_eq!(prefix.text, t.decode(&invalid, true));
    assert!(prefix.pending_token_ids.is_empty());
    assert!(prefix.text.contains('\u{fffd}'));
    let special = t.decode_complete_prefix(&[1, 2, 3, 4, 2, 5, 10], true);
    assert_eq!(special.text, "prefix睡");
    assert_eq!(special.pending_token_ids, vec![2, 5, 10]);
    assert!(
        t.decode_complete_prefix(&[1, 2, 3, 4, 10], true)
            .pending_token_ids
            .is_empty()
    );
}
