s = """
diff --git a/libavformat/http.c b/libavformat/http.c
index 510b233..aa80e31 100644
--- a/libavformat/http.c
+++ b/libavformat/http.c
@@ -1449,12 +1449,18 @@ static int http_read_stream(URLContext *h, uint8_t *buf, int size)
         return http_buf_read_compressed(h, buf, size);
 #endif /* CONFIG_ZLIB */
     read_ret = http_buf_read(h, buf, size);
-    while ((read_ret  < 0           && s->reconnect        && (!h->is_streamed || s->reconnect_streamed) && s->filesize > 0 && s->off < s->filesize)
-        || (read_ret == AVERROR_EOF && s->reconnect_at_eof && (!h->is_streamed || s->reconnect_streamed))) {
+    while (read_ret < 0) {
         uint64_t target = h->is_streamed ? 0 : s->off;
 
         if (read_ret == AVERROR_EXIT)
-            return read_ret;
+            break;
+
+        if (h->is_streamed && !s->reconnect_streamed)
+            break;
+
+        if (!(s->reconnect && s->filesize > 0 && s->off < s->filesize) &&
+            !(s->reconnect_at_eof && read_ret == AVERROR_EOF))
+            break;
 
         if (reconnect_delay > s->reconnect_delay_max)
             return AVERROR(EIO);
"""

lines = s.split('\n')
added_lines = []
deleted_lines = []

for line in lines:
    if line.startswith('+') and not line.startswith('+++'):
        added_lines.append(line[1:])
    elif line.startswith('-') and not line.startswith('---'):
        deleted_lines.append(line[1:])

print("新增的代码:")
print(' '.join(added_lines).strip())
print("删除的代码:")
print(' '.join(deleted_lines).strip())