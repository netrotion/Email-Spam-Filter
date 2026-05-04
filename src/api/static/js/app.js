(() => {
    document.documentElement.classList.add("js-ready");

    const onReady = (callback) => {
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", callback, { once: true });
            return;
        }
        callback();
    };

    const csvCell = (value) => `"${String(value ?? "").replace(/"/g, "\"\"")}"`;

    const bindLoadingState = () => {
        document.querySelectorAll("form button[type='submit'][data-loading-text]").forEach((button) => {
            const defaultText = button.textContent;
            button.closest("form")?.addEventListener("submit", () => {
                button.textContent = button.dataset.loadingText || defaultText;
                button.disabled = true;
            });
        });
    };

    const bindBatchDownload = () => {
        const button = document.getElementById("download-batch-results");
        const tableBody = document.getElementById("batch-results-body");
        if (!button || !tableBody) {
            return;
        }

        button.addEventListener("click", () => {
            const rows = [...tableBody.querySelectorAll("tr")];
            if (!rows.length) {
                return;
            }

            const records = rows.map((row) => {
                const exportRow = [
                    row.dataset.exportRow || (row.children[0]?.textContent || "").trim(),
                    row.dataset.exportText || "",
                    row.dataset.exportPreview || (row.children[1]?.textContent || "").trim(),
                    row.dataset.exportLabel || (row.children[2]?.textContent || "").trim(),
                    row.dataset.exportConfidence || (row.children[3]?.textContent || "").trim(),
                    row.dataset.exportSpamProbability || "",
                    row.dataset.exportHamProbability || "",
                    row.dataset.exportTopSignals || "",
                ];
                return exportRow.map(csvCell).join(",");
            });

            const csvContent = [
                "row,text,preview,label,confidence,spam_probability,ham_probability,top_signals",
                ...records,
            ].join("\n");
            const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
            const objectUrl = URL.createObjectURL(blob);
            const anchor = document.createElement("a");
            anchor.href = objectUrl;
            anchor.download = "batch_predictions.csv";
            anchor.click();
            URL.revokeObjectURL(objectUrl);
        });
    };

    onReady(() => {
        bindLoadingState();
        bindBatchDownload();
    });
})();
