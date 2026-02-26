import numpy as np

def run_train_test(training_input, testing_input):
    """
    Implements MP1 three-class "basic linear classifier" using centroids + pairwise
    discriminant functions, with hierarchical testing and averaged metrics.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values with keys:
        "tpr", "fpr", "error_rate", "accuracy", "precision"
    """

    # I used Chatgbt to help outline a code then correct any coding mistakes
    # Parse data into class-specific matrices (A, B, C)
    def parse_data(input_list):
        """
        input_list[0] is the header: [3, nA, nB, nC]
        remaining rows are feature vectors in order: A then B then C
        """
        header = input_list[0]
        n_a = int(header[1])
        n_b = int(header[2])
        n_c = int(header[3])

        data = np.asarray(input_list[1:], dtype=float)  # shape: (n_a+n_b+n_c, d)

        X_a = data[0:n_a]
        X_b = data[n_a:n_a + n_b]
        X_c = data[n_a + n_b:n_a + n_b + n_c]

        return X_a, X_b, X_c


    # Training = compute centroids for each class
    train_a, train_b, train_c = parse_data(training_input)

    centroid_a = np.mean(train_a, axis=0)
    centroid_b = np.mean(train_b, axis=0)
    centroid_c = np.mean(train_c, axis=0)

    # Step 3: Pairwise discriminant function (linear boundary)

    def classify_pair(x, c1, c2):
        """
        Returns True if classified as class-1 (centroid c1),
        False if classified as class-2 (centroid c2).
        Ties go to class-1.
        """
        m = 0.5 * (c1 + c2)
        w = (c2 - c1)

        score = np.dot(x - m, w)

        # score < 0 => class 1 side; score > 0 => class 2 side
        # tie => class 1 (priority)
        return score <= 0.0

    # Testing with the required hierarchical procedure
    test_a, test_b, test_c = parse_data(testing_input)

    # Build a combined test list of (x, actual_label)
    # label encoding: 0=A, 1=B, 2=C
    test_samples = []
    for x in test_a:
        test_samples.append((x, 0))
    for x in test_b:
        test_samples.append((x, 1))
    for x in test_c:
        test_samples.append((x, 2))

    # Confusion matrix: rows = predicted, cols = actual
    # index 0=A, 1=B, 2=C
    confusion = np.zeros((3, 3), dtype=int)

    for x, actual in test_samples:
        # First: A vs B
        is_A = classify_pair(x, centroid_a, centroid_b)  # True => A, False => B

        if is_A:
            # Now: A vs C
            is_A2 = classify_pair(x, centroid_a, centroid_c)  # True => A, False => C
            pred = 0 if is_A2 else 2
        else:
            # Now: B vs C
            is_B2 = classify_pair(x, centroid_b, centroid_c)  # True => B, False => C
            pred = 1 if is_B2 else 2

        confusion[pred, actual] += 1

    # Compute metrics per class using TP/FN/FP/TN from 3-class confusion
    # Then average across classes A,B,C

    total = len(test_samples)

    sum_tpr = 0.0
    sum_fpr = 0.0
    sum_error = 0.0
    sum_acc = 0.0
    sum_prec = 0.0

    for i in range(3):
        tp = confusion[i, i]
        fp = np.sum(confusion[i, :]) - tp          # row i excluding diagonal
        fn = np.sum(confusion[:, i]) - tp          # col i excluding diagonal
        tn = total - (tp + fp + fn)

        # TPR = TP / (TP + FN)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # FPR = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Error rate = (FP + FN) / Total
        error_rate = (fp + fn) / total

        # Accuracy = (TP + TN) / Total
        accuracy = (tp + tn) / total

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        sum_tpr += tpr
        sum_fpr += fpr
        sum_error += error_rate
        sum_acc += accuracy
        sum_prec += precision

    return {
        "tpr": sum_tpr / 3.0,
        "fpr": sum_fpr / 3.0,
        "error_rate": sum_error / 3.0,
        "accuracy": sum_acc / 3.0,
        "precision": sum_prec / 3.0
    }
