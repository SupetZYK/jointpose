#pragma once

/** Gray pixel difference feature and feature sampler */
namespace jp
{
    /**
     * @brief Random forest feature that calculate the differences of two pixel probes around a center pixel.
     */
    class FeatureDAGRAY
    {
    public:
        /**
         * @brief Default contructor.
         */
        FeatureDAGRAY() : off1_x(0), off1_y(0), off2_x(0), off2_y(0), training(false)
        {
        }

        /**
         * @brief Constructs a feature.
         *
         * @param off1_x X component of the offset vector of pixel probe 1.
         * @param off1_y Y component of the offset vector of pixel probe 1.
         * @param off2_x X component of the offset vector of pixel probe 2.
         * @param off2_y Y component of the offset vector of pixel probe 2.
         * @param training Does the feature operate in training mode? (Simulation of noisy responses)
         */
        FeatureDABGR(int off1_x, int off1_y, int off2_x, int off2_y, bool training) :
            off1_x(off1_x), off1_y(off1_y),
            off2_x(off2_x), off2_y(off2_y),
            training(training)
        {
        }

        /**
         * @brief Returns a feature type ID.
         *
         * @return uchar Feature type ID.
         */
        uchar getType() const { return 11; }

        /**
         * @brief Computes the feature value at the given center pixel. Feature offsets are scaled by the given scale factor.
         *
         * @param x X component of the center pixel.
         * @param y Y component of the center pixel.
         * @param scale Feature offsets are scaled by this factor.
         * @param data Input frame.
         * @return double Feature response.
         */
        double computeResponse(int x, int y, float scale, const jp::img_data_t& data) const
        {
            // scale and clamp offset vectors
            FeaturePoints fP = getFeaturePoints(x, y, off1_x, off1_y, off2_x, off2_y, scale, data.seg.cols, data.seg.rows);

            double val1, val2;

             //read out pixel probe 1
            if(data.seg(fP.y1, fP.x1))
                val1 = 0.4*(double) data.colorData(fP.y1, fP.x1)(0) + 0.3*(double) data.colorData(fP.y1, fP.x1)(1) + 0.3*(double) data.colorData(fP.y1, fP.x1)(2);
            else
                val1 = irand(0, 256); // return a random color outside ground truth segmentation

             // read out pixel probe 2
            if(data.seg(fP.y2, fP.x2))
                val2 = 0.4*(double) data.colorData(fP.y2, fP.x2)(0) + 0.3*(double) data.colorData(fP.y2, fP.x2)(1) + 0.3*(double) data.colorData(fP.y2, fP.x2)(2);
            else
                val2 = irand(0, 256); // return a random color outside ground truth segmentation

            return val1 - val2; // feature response is difference of pixel probe values
        }

        /**
         * @brief Calculates the feature response and compares it to the feature threshold.
         *
         * @param x X component of the center pixel.
         * @param y Y component of the center pixel.
         * @param scale Feature offsets are scaled by this factor.
         * @param data Input frame.
         * @return bool False if feature response if below threshold.
         */
        bool operator()(int x, int y, float scale, const jp::img_data_t& data) const
        {
            double resp = computeResponse(x, y, scale, data);
            return resp >= thresh;
        }

        /**
         * @brief Set the feature threshold.
         *
         * @return void
         */
        void setThreshold(double thresh) { this->thresh = thresh; }

        /**
         * @brief Print the parameters of the feature to the console.
         *
         * @return void
         */
        void print() const
        {
            std::cout << "Gray Feature (x1: " << off1_x << ", y1: " << off1_y
                << ", x2: " << off2_x << ", y2: " << off2_y
                << ", t: " << thresh << ")" << std::endl;
        }

        /**
         * @brief Write the feature to the given file.
         *
         * @param file File to write to.
         * @return void
         */
        void store(std::ofstream& file) const
        {
            //write(file, channel1);
            //write(file, channel2);
            write(file, off1_x);
            write(file, off1_y);
            write(file, off2_x);
            write(file, off2_y);
            write(file, thresh);
        }

        /**
         * @brief Read the feature from the given file.
         *
         * @param file File to read from.
         * @return void
         */
        void restore(std::ifstream& file)
        {
            //read(file, channel1);
            //read(file, channel2);
            read(file, off1_x);
            read(file, off1_y);
            read(file, off2_x);
            read(file, off2_y);
            read(file, thresh);
            training = false;
        }

    private:
        //int channel1; // RGB channel number of the first pixel probe
        //int channel2; // RGB channel number of the second pixel probe
        int off1_x, off1_y; // offset vector of the first pixel probe
        int off2_x, off2_y; // offset vector of the second pixel probe
        double thresh; // feature threshold
        bool training; // does the feature operate in training mode? (can be used to simulate noise during training)
    };

    /**
     * @brief Write a feature to the given file.
     *
     * @param file File to write to.
     * @param feature Feature to write.
     * @return void
     */
    template<>
    void write(std::ofstream& file, const FeatureDAGRAY& feature);

    /**
     * @brief Read a feature from the given file.
     *
     * @param file File to read from.
     * @param feature Feature read from.
     * @return void
     */
    template<>
    void read(std::ifstream& file, FeatureDAGRAY& feature);

    /**
     * @brief Class to randomly sample RGB pixel difference features.
     */
    class FeatureSamplerDAGRAY
    {
    public:
        typedef FeatureDAGRAY feature_t;

        /**
         * @brief Constructor. The (x,y) offsets for the feature tests are sampled from a uniform distribution
         * from -off_max to off_max.
         *
         * @param off_max Maximally allowed feature offset.
         */
        FeatureSamplerDAGRAY(int off_max) : off_max(off_max)
        {
        }

        /**
          * @brief Generate one random feature.
          *
          * @return jp::FeatureSamplerDABGR::feature_t Random feature.
          */
        feature_t sampleFeature() const
        {
            return feature_t(getOffset(), getOffset(), getOffset(), getOffset(), true);
        }

        /**
         * @brief Create a number of IDENTICAL feautures.
         *
         * Feature offsets and RGB channels for the two pixel probes will be choosen randomly once and set for all features.
         *
         * @param count How meany features to generate?
         * @return std::vector< feature_t > List of identical features.
         */
        std::vector<feature_t> sampleFeatures(unsigned count) const
        {
            // create number of thresholds of identical features
            int offset1 = getOffset();
            int offset2 = getOffset();
            int offset3 = getOffset();
            int offset4 = getOffset();

            std::vector<feature_t> features;
            for(unsigned i = 0; i < count; i++)
            {
                features.push_back(feature_t(offset1, offset2, offset3, offset4, true));
            }

            return features;
        }

    private:
        int off_max; // maximally allowed offset vector

        /**
         * @brief Returns a random pixel offset.
         *
         * @return int Offset.
         */
        int getOffset() const { return irand(-off_max, off_max + 1); }
    };
}
