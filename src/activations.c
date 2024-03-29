#include <assert.h>
#include <math.h>
#include "uwnet.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i*m.cols + j];
            if(a == LOGISTIC){
                // TODO
                m.data[i*m.cols + j] = 1 / 1 + expf(-x);
            } else if (a == RELU){
                // TODO
                m.data[i*m.cols + j] = fmaxf(0, x);
            } else if (a == LRELU){
                if (x > 0) {
                    m.data[i*m.cols + j] = x;
                } else {
                    m.data[i*m.cols + j] = 0.1 * x;
                }
            } else if (a == SOFTMAX){
                m.data[i*m.cols + j] = expf(x);
            }
            sum += m.data[i*m.cols + j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
            for (j = 0; j < m.cols; j++) {
                m.data[i*m.cols + j] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i*m.cols + j];
            // TODO: multiply the correct element of d by the gradient
            if(a == LOGISTIC){
                d.data[i*m.cols + j] *= x * (1-x);
            } else if (a == RELU && x < 0){
                d.data[i*m.cols + j] = 0;
            } else if (a == LRELU && x < 0){
                d.data[i*m.cols + j] *= 0.1;
            }
        }
    }
}
