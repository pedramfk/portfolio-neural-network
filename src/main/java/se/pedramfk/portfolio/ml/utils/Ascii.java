package se.pedramfk.portfolio.ml.utils;

public class Ascii {

    public static final char[] repeatChar(char c, int n) {
        final char[] chars = new char[n];
        for (int i = 0; i < n; i++) chars[i] = c;
        return chars;
    }

    public static class MatrixAscii {

        public static final int DEFAULT_CELL_WIDTH = 11;
        public static final int DEFAULT_DECIMALS = 1;
        public static final double DEFAULT_EXP_FORMAT_THRESHOLD = 10000.0;

        public static final char DEFAULT_NEW_LINE = '\n';
        public static final char DEFAULT_LEFT_PAD = ' ';

        public static final char MID_INTER_BORDER = '┼';
        public static final char TOP_INTER_BORDER = '┬';
        public static final char BOTTOM_INTER_BORDER = '┴';
        public static final char LEFT_INTER_BORDER = '├';
        public static final char RIGHT_INTER_BORDER = '┤';

        public static final char TOP_LEFT_BORDER = '┌';
        public static final char TOP_RIGHT_BORDER = '┐';
        public static final char BOTTOM_LEFT_BORDER = '└';
        public static final char BOTTOM_RIGHT_BORDER = '┘';

        public static final char HORIZONTAL_LINE = '─';
        public static final char VERTICAL_LINE = '│';

        public static String getString(Matrix matrix, String label, int r) {

            StringBuilder sb = new StringBuilder();

            final int boxWidth = matrix.cols * DEFAULT_CELL_WIDTH + (matrix.cols + 2);

            // Header
            final int headerLeftPad = label.length() < boxWidth ? Math.round((boxWidth - label.length()) / 2) : 0;

            sb.append(DEFAULT_NEW_LINE);
            sb.append(DEFAULT_LEFT_PAD);
            sb.append(repeatChar(DEFAULT_LEFT_PAD, headerLeftPad));
            sb.append(label);

            // Top Border
            sb.append(DEFAULT_NEW_LINE);
            sb.append(DEFAULT_LEFT_PAD);
            sb.append(TOP_LEFT_BORDER);
            for (int i = 0; i < matrix.cols; i++) {
                sb.append(repeatChar(HORIZONTAL_LINE, DEFAULT_CELL_WIDTH));
                sb.append(i < matrix.cols - 1 ? TOP_INTER_BORDER : TOP_RIGHT_BORDER);
            }

            // Intermediate Borders
            for (int row = 0; row < matrix.rows; row++) {

                sb.append(DEFAULT_NEW_LINE);
                sb.append(DEFAULT_LEFT_PAD);
                sb.append(VERTICAL_LINE);

                for (int col = 0; col < matrix.cols; col++) {

                    final boolean expFormat = matrix.get(row, col) >= DEFAULT_EXP_FORMAT_THRESHOLD ? true : false;
                    final String formatString = String.format("%s.%d%s", "%", r, expFormat ? "e" : "f");
                    final String cell = String.format(formatString, matrix.get(row, col));

                    final int cellLeftPad = Math.round((DEFAULT_CELL_WIDTH - cell.length()) / 2);
                    final int cellRightPad = DEFAULT_CELL_WIDTH - cell.length() - cellLeftPad;

                    sb.append(repeatChar(DEFAULT_LEFT_PAD, cellLeftPad));
                    sb.append(cell);
                    sb.append(repeatChar(DEFAULT_LEFT_PAD, cellRightPad));
                    sb.append(VERTICAL_LINE);

                }

                if (row < matrix.rows - 1) {

                    sb.append(DEFAULT_NEW_LINE);
                    sb.append(DEFAULT_LEFT_PAD);
                    sb.append(LEFT_INTER_BORDER);
                    for (int i = 0; i < matrix.cols; i++) {
                        sb.append(repeatChar(HORIZONTAL_LINE, DEFAULT_CELL_WIDTH));
                        sb.append(i < matrix.cols - 1 ? MID_INTER_BORDER : RIGHT_INTER_BORDER);
                    }

                }

            }

            // Bottom Border
            sb.append(DEFAULT_NEW_LINE);
            sb.append(DEFAULT_LEFT_PAD);
            sb.append(BOTTOM_LEFT_BORDER);
            for (int i = 0; i < matrix.cols; i++) {
                sb.append(repeatChar(HORIZONTAL_LINE, DEFAULT_CELL_WIDTH));
                sb.append(i < matrix.cols - 1 ? BOTTOM_INTER_BORDER : BOTTOM_RIGHT_BORDER);
            }

            // Footnote
            final String footnote = String.format("[%d × %d]", matrix.rows, matrix.cols);
            sb.append(DEFAULT_NEW_LINE);
            sb.append(DEFAULT_LEFT_PAD);
            sb.append(footnote);

            return sb.toString();

        }

        public static String getString(Matrix matrix, int r) {
            return getString(matrix, matrix.label == null ? "" : matrix.label, r);
        }

        public static String getString(Matrix matrix, String label) {
            return getString(matrix, label, DEFAULT_DECIMALS);
        }

        public static String getString(Matrix matrix) {
            return getString(matrix, DEFAULT_DECIMALS);
        }

    }

    public static final void main(String[] args) {

        Matrix w = Matrix.fromArray(new double[][] {
            {2012.9821, 119.1, 400.1}, 
            {191.12111, 18.121, 232.2}, 
            {-12, 12, 90}
        });

        w.label = "Layer 1 - W";

        System.out.println(MatrixAscii.getString(w));

    }
    
}
