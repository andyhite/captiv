css = """
.main {
  flex-shrink: 0;
  flex-grow: 1;
}

.main .body .gallery .thumbnails {
  padding-left: 100px;
  padding-right: 10px;
}

.main .body .gallery .gallery-container {
  height: 100%;
  overflow-y: auto !important;
  scrollbar-color: #888 transparent;
  scrollbar-width: thin;
}

.main .body .gallery .gallery-container::-webkit-scrollbar {
  width: 8px;
}

.main .body .gallery .gallery-container::-webkit-scrollbar-track {
  background: transparent; /* Chrome/Safari/Edge */
}

.main .body .gallery .gallery-container::-webkit-scrollbar-thumb {
  background-color: #888;
  border-radius: 4px;
}

.main .body .gallery .gallery-container .fixed-height {
  max-height: 0;
}

.main .body .gallery .gallery-container .grid-wrap {
  overflow: visible;
}
"""
