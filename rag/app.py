import os

from flask import Flask, request, jsonify
from query import answer_question, search_pdf, DOCUMENT_CATALOG, MIN_RELEVANCE_SCORE

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json(force=True)

    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Missing required field: question'}), 400

    k = int(data.get('k', 10))
    use_llm = bool(data.get('use_llm', True))
    doc_filter = data.get('doc_filter') or None
    min_score = float(data.get('min_score', MIN_RELEVANCE_SCORE))
    history = data.get('history') or None  # list of {role, content} dicts

    import query as q_module
    original = q_module.MIN_RELEVANCE_SCORE
    q_module.MIN_RELEVANCE_SCORE = min_score
    try:
        result = answer_question(question, k=k, use_llm=use_llm, doc_filter=doc_filter, history=history)
    finally:
        q_module.MIN_RELEVANCE_SCORE = original

    return jsonify(result)


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json(force=True)

    query_text = data.get('query', '').strip()
    if not query_text:
        return jsonify({'error': 'Missing required field: query'}), 400

    k = int(data.get('k', 10))
    use_hybrid = bool(data.get('use_hybrid', True))
    doc_filter = data.get('doc_filter') or None

    results = search_pdf(query_text, k=k, use_hybrid=use_hybrid, doc_filter=doc_filter)
    return jsonify({'results': results})



@app.route('/catalog', methods=['GET'])
def catalog():
    return jsonify({'documents': DOCUMENT_CATALOG})


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
