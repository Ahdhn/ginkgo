#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#include <limits.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace Neon {
namespace HelpTools {

class NormalEstimator
{
   private:
    uint64_t    m_nSamples{0};
    double      m_average{0.0};
    double      m_stdDeviation{0.0};
    std::string m_measurementUnit{"Unknown"};
    std::string m_label{"Unknown"};

   private:
    NormalEstimator(uint64_t           Nsamples,
                    double             average,
                    double             standardDeviation,
                    const std::string& measuramentUnit,
                    const std::string& label)
    : m_nSamples(Nsamples), m_average(average),
      m_stdDeviation(standardDeviation), m_measurementUnit(measuramentUnit),
      m_label(label){}


   public:
    /*
     * Create an object that keep track streaming average and standard
     * deviation.
     *
     * \param[in] unit  : measurement unit associated with the samples.
     * \param[in] label : name of the measured entity.
     */
    NormalEstimator(std::string label, std::string unit)
    : m_measurementUnit(std::move(unit)), m_label(std::move(label)){}

    NormalEstimator() = default;
    NormalEstimator(const NormalEstimator&) = default;
    NormalEstimator(NormalEstimator&&) = default;
    NormalEstimator& operator=(const NormalEstimator&) = default;
    NormalEstimator& operator=(NormalEstimator&&) = default;
    ~NormalEstimator() = default;

    /*
     * Add a sample
     *
     * \param[in] x  : value of the sample
     */
    void addSample(double x){
        ++m_nSamples;
        double nextM = m_average + (x - m_average) / m_nSamples;
        m_stdDeviation += (x - m_average) * (x - nextM);
        m_average = nextM;
    }

    /*
     * Return number of samples
     *
     * \return : number of accumulated samples
     */
    uint64_t numSample() const{
       return m_nSamples;
    }

    /*
     * Returns the mean over the accumulated samples.
     *
     * \return  : mean
     */
    double mean() const{
        return m_average;
    }
    /*
     * Returns the variance over the accumulated samples.
     *
     * \return  : variance
     */
    double variance() const
    {
        return m_nSamples > 1 ? m_stdDeviation / static_cast<double>(m_nSamples) : 0.0;
    }
    /*
     * Returns the unbiased variance over the accumulated samples.
     *
     * \return  : unbiased variance
     */
    double varianceUnbiased() const
    {
        return m_nSamples > 1 ? m_stdDeviation / static_cast<double>(m_nSamples - 1) : 0.0;
    }
    /*
     * Returns the standard deviation over the accumulated samples.
     *
     * \return  : standard deviation
     */
    double standardDeviation() const
    {
        return std::sqrt(this->variance());
    }
    /*
     * Returns the unbiased standard deviation over the accumulated samples.
     *
     * \return  : unbiased standard deviation
     */
    double standardDeviationUnbiased() const
    {
        return std::sqrt(this->varianceUnbiased());
    }

    std::string get_label() const
    {
        return m_label;
    }
    std::string get_unit() const
    {
        return m_measurementUnit;
    }

    void reset()
    {
        m_nSamples = 0;
        m_average = 0;
        m_stdDeviation = 0;
    }
};

}  // namespace HelpTools


class PerfRecord_t
{
    static const bool number = true;
    static const bool addDefaultInfo = true;

   public:
    /*
     *	Structure to store information of static attributes.
     * 	The information is stored only as strings.
     */
    class staticAtt_t
    {
       private:
        const std::string m_name;
        const std::string m_value;
        const std::string m_measureUnit;
        const bool        m_isNumber;


       public:
        staticAtt_t(const std::string& _name,
                    const std::string& _value,
                    const bool         _isNumber,
                    const std::string& _measureUnit = std::string(""))
            : m_name(_name), m_value(_value), m_measureUnit(_measureUnit),
              m_isNumber(_isNumber)
        {
        }

        staticAtt_t& operator=(const staticAtt_t& other)
        {
            new (this) staticAtt_t(other.m_name, other.m_value,
                                   other.m_isNumber, other.m_measureUnit);
            return *this;
        }

        std::string getName() const
        {
            return m_name;
        }
        std::string getValue() const
        {
            return m_value;
        }
        std::string getUnit() const
        {
            return m_measureUnit;
        }
        bool number() const
        {
            return m_isNumber;
        }
    };

    struct dynamicAtt_t
    {
        HelpTools::NormalEstimator ne;

        dynamicAtt_t(const std::string& _name, const std::string& _measureUnit)
            : ne(_name, _measureUnit)
        {
        }

        std::string getName() const
        {
            return ne.get_label();
        }

        std::string getUnit() const
        {
            return ne.get_unit();
        }
    };

   private:
    std::vector<staticAtt_t>  m_stat;
    std::vector<dynamicAtt_t> m_dyn;

   public:
    PerfRecord_t(const std::string& recordName)
    {
        this->addStaticAtt("recordName", recordName, !number);
    }

    void addStaticAtt(const std::string& name,
                      const std::string& val,
                      const bool         isNumber,
                      const std::string& mUnit = std::string(""))
    {
        m_stat.emplace_back(staticAtt_t(name, val, isNumber, mUnit));
    }

    int addDynamicAtt(const std::string& name, const std::string& unit)
    {
        m_dyn.emplace_back(dynamicAtt_t(name, unit));
        return static_cast<int>(m_dyn.size() - 1);
    }

    void updateDynamicAtt(const int id, double val)
    {
        if (static_cast<size_t>(id) >= m_dyn.size()) {
            std::cout << "Unable to add valued for ID " + std::to_string(id) +
                             ".\n";
            exit(EXIT_FAILURE);
        }
        m_dyn[id].ne.addSample(val);
    }

    /**
     *	Update the value associated with an dynamic attribute. The value is
     *average with the previous values. The attribute compute streaming average
     *and standard variation of all the values.
     *
     *	@param[in]	name	Name of the target attribute.
     *	@param[in]	val		New Value to add for averaging.
     *	@throw				Dreamutil_error if the name does not match any registered
     *dynamic attribute.
     *
     */
    void updateDynamicAtt(const std::string& name, const double val)
    {
        for (auto&& rec : m_dyn) {
            if (name == rec.getName()) {
                rec.ne.addSample(val);
                return;
            }
        }
        std::cout << "Unable to add valued for variable mane " + name + ".\n";
        exit(EXIT_FAILURE);
    }

    std::string jsonStaticAtt(const int indentLevel, const staticAtt_t& att)
    {
        const std::string QT("\"");
        const std::string BLOCK_START(": {");
        const std::string NL("\n");
        const std::string TAB("     ");
        const std::string EQ(" : ");
        const std::string COMMA(" , ");

        std::string json("");

        std::string INDENT;

        for (int i = 0; i < indentLevel; ++i) {
            INDENT += TAB;
        }

        INDENT += INDENT + TAB;
        json += INDENT + std::string("{") + NL;
        json += INDENT + QT + std::string("name") + QT + EQ + QT +
                att.getName() + QT + COMMA + NL;

        if (att.number()) {
            json += INDENT + QT + std::string("value") + QT + EQ + QT +
                    att.getValue() + QT + COMMA + NL;
            json += INDENT + QT + std::string("unit") + QT + EQ + QT +
                    att.getUnit() + QT + NL;
        } else {
            json += INDENT + QT + std::string("value") + QT + EQ + QT +
                    att.getValue() + QT + NL;
        }
        json += INDENT + std::string("}");

        return json;
    }

    std::string jsonDynamicAtt(int indent_level, dynamicAtt_t& att)
    {
        const std::string QT("\"");
        const std::string BLOCK_START(": {");
        const std::string NL("\n");
        const std::string TAB("     ");
        const std::string EQ(" : ");
        const std::string COMMA(" , ");

        std::string json("");

        std::string INDENT;
        for (int i = 0; i < indent_level; ++i) {
            INDENT += TAB;
        }

        INDENT += INDENT + TAB;
        json += INDENT + std::string("{") + NL;

        json += INDENT + QT + std::string("name") + QT + EQ + QT +
                att.getName() + QT + COMMA + NL;
        json += INDENT + QT + std::string("unit") + QT + EQ + QT +
                att.getUnit() + QT + COMMA + NL;
        json += INDENT + QT + std::string("numSample") + QT + EQ +
                std::to_string(att.ne.numSample()) + COMMA + NL;
        json += INDENT + QT + std::string("mean") + QT + EQ +
                std::to_string(att.ne.mean()) + COMMA + NL;
        json += INDENT + QT + std::string("variance") + QT + EQ +
                std::to_string(att.ne.variance()) + COMMA + NL;
        json += INDENT + QT + std::string("varianceUnbiased") + QT + EQ +
                std::to_string(att.ne.varianceUnbiased()) + COMMA + NL;
        json += INDENT + QT + std::string("standardDeviation") + QT + EQ +
                std::to_string(att.ne.standardDeviation()) + COMMA + NL;
        json += INDENT + QT + std::string("standardDeviationUnbiased") + QT +
                EQ + std::to_string(att.ne.standardDeviationUnbiased()) + NL;

        json += INDENT + std::string(" }");
        return json;
    }

    std::string json(int initialIndentLevel)
    {
        const std::string QT("\"");
        const std::string BLOCK_START(": {");
        const std::string NL("\n");
        const std::string TAB("     ");
        const std::string EQ(" : ");
        const std::string COMMA(" , ");

        int&        indent_level = initialIndentLevel;
        std::string json;
        std::string INDENT;
        for (int i = 0; i < indent_level; ++i) {
            INDENT += TAB;
        }

        json += INDENT + std::string("{") + NL;
        INDENT += INDENT + TAB;
        json += INDENT + QT + std::string("StaticAttribute") + QT +
                std::string(": [") + NL;
        for (size_t i = 0; i < m_stat.size(); i++) {
            json += jsonStaticAtt(indent_level + 1, m_stat[i]);
            if ((i != m_stat.size() - 1)) {
                json += COMMA;
            } else {
                json += NL + INDENT + std::string("]");
                if (!m_dyn.empty()) {
                    json += COMMA + NL;
                }
            }
            json += NL;
        }
        json += INDENT + QT + std::string("DynamicAttribute") + QT +
                std::string(": [") + NL;
        for (size_t i = 0; i < m_dyn.size(); i++) {
            json += jsonDynamicAtt(indent_level + 1, m_dyn[i]);
            if (i != m_dyn.size() - 1) {
                json += COMMA;
            } else {
                json += NL + INDENT + std::string("]");
            }
            json += NL;
        }
        INDENT = std::string("");
        for (int i = 0; i < indent_level; ++i) {
            INDENT += TAB;
        }
        json += INDENT + std::string("}");
        return json;
    }
};

class PerfKeeper_t
{
   public:
    using recId_t = int;
    using dynAttId_t = int;
    static const bool addAsNumber = true;
    static const bool addDefaultInfo = true;

   private:
    std::vector<PerfRecord_t> m_fields;

    std::string getHostName()
    {
#ifndef _WIN32
        char hostname[300];
        gethostname(hostname, 300 - 1);
        hostname[300 - 1] = '\0';
        return std::string(hostname);
#else
        return std::string("Not supported yet.");
#endif
    }

   public:
    /*
     * Constructor for the perfKeeper
     */
    PerfKeeper_t() = default;

    void addDefaultInfoToRecord(const recId_t recId)
    {
        {
            using namespace std;

            {
                int ompMaxThreads = 0;
#if defined(_OPENMP)
                ompMaxThreads = omp_get_max_threads();
                this->addStaticAtt(recId, "NEON_OMP", "ON", addAsNumber,
                                   "Status");
#else
                this->addStaticAtt(recId, "NEON_OMP", "OFF", addAsNumber,
                                   "Status");
                ompMaxThreads = 1;

#endif

                this->addStaticAtt(recId, "NEON_NumOmpThreads",
                                   to_string(ompMaxThreads), addAsNumber,
                                   "Number of Threads");
            }
            std::chrono::time_point<std::chrono::system_clock> nowInstant;
            nowInstant = std::chrono::system_clock::now();
            std::time_t nowTime =
                std::chrono::system_clock::to_time_t(nowInstant);

            std::string ctimeString(std::ctime(&nowTime));

            string::size_type pos = 0;
            pos = ctimeString.find("\r\n", pos);
            if (string::npos != pos) {
                ctimeString.erase(pos, 2);
            }
            pos = 0;
            pos = ctimeString.find('\n', pos);
            if (string::npos != pos) {
                ctimeString.erase(pos, 1);
            }
            this->addStaticAtt(recId, "NEON_TimeRef", ctimeString,
                               !addAsNumber);
        }
    }

    /**
     * Method to add a performance record.
     * @param[in] recordName: Name associated to the record.
     * @param[in] doAddDefaultInfo: if true some static attributes are added by
     * default (omp threads, compiler info)
     * @return A record identifier.
     */
    recId_t addRecord(const std::string& recordName, bool doAddDefaultInfo)
    {
        m_fields.emplace_back(PerfRecord_t(recordName));
        recId_t recId = recId_t(m_fields.size() - 1);
        this->addStaticAtt(recId, "NEON_RECORD_STORAGE_TYPE", "FromRuntime",
                           !addAsNumber);
        if (doAddDefaultInfo == addDefaultInfo) {
            addDefaultInfoToRecord(recId);
        }
        return recId;
    }

    /**
     * Method to add a static attribute to a record
     * @param[in] id:   record id
     * @param[in] name: name of the attribute
     * @param[in] val: value of the attribute
     * @param[in] isNumber: if true (or number), the value of the attribute is a
     * number
     * @param[in] mUnit: optional string for the unit used to express the value.
     */
    void addStaticAtt(const recId_t      id,
                      const std::string& name,
                      const std::string& val,
                      const bool         isNumber,
                      const std::string& mUnit = std::string(""))
    {
        m_fields[id].addStaticAtt(name, val, isNumber, mUnit);
    }

    /**
     * Method to add a dynamic attribute
     * @param[in] recId: record id
     * @param[in] name: name of the attribute
     * @param[in] unit: optional string for the unit used to express the value.
     */
    dynAttId_t addDynamicAtt(const recId_t      recId,
                             const std::string& name,
                             const std::string& unit)
    {
        return m_fields[recId].addDynamicAtt(name, unit);
    }

    /**
     * Method to add a new sample to a dynamic attribute.
     * @param[in] recId: record id
     * @param[in] attId: attribute Id
     * @param[in] val: new value.
     */
    void updateDynamicAtt(const recId_t recId, dynAttId_t attId, double val)
    {
        return m_fields[recId].updateDynamicAtt(attId, val);
    }

    /**
     * Method to add a new sample to a dynamic attribute.
     * @param[in] recId: record id
     * @param[in] name: attribute name
     * @param[in] val: new value.
     */
    void updateDynamicAtt(const recId_t      recId,
                          const std::string& name,
                          const double       val)
    {
        return m_fields[recId].updateDynamicAtt(name, val);
    }

    /**
     * Method to save all the record into a Json file. The method will add the
     * extension "..perfKeeper" if it not included into the file name.
     *
     * @param[in] fileName: file name.
     *
     */
    void saveJson(std::string fileName = std::string("test.perfKeeper"))
    {
        {  /// Automatically add suffix
            using namespace std;
            using S = std::string;
            size_t size = fileName.size();
            size_t sufixSize = S(".json").size();
            bool   extentionToBeAdded = false;
            if (size < sufixSize) {
                extentionToBeAdded = true;
            }
            if (size < 1 + sufixSize) {
                extentionToBeAdded = true;
            } else {
                size_t startingPoint = size - 1 - sufixSize;
                if (fileName.substr(startingPoint, sufixSize) != S(".json")) {
                    extentionToBeAdded = true;
                }
            }
            if (extentionToBeAdded) {
                fileName = fileName + S(".json");
            }
        }
        const unsigned int initialIndentLevel = 1;
        std::ofstream      theFile;
        theFile.open(fileName);
        if (!theFile) {
            throw std::runtime_error(
                "[PerfKeeper_t] ERROR: Unable to open file: " + fileName);
        }
        theFile << get_JSON(initialIndentLevel) << std::endl;
    }

   private:
    std::string get_JSON(int initialIndentLevel)
    {
        int&        indent_level = initialIndentLevel;
        std::string json;
        std::string QT = std::string("\"");
        std::string BLOCK_START = std::string(": {");
        std::string NL = std::string("\n");
        std::string TAB = std::string("     ");
        std::string INDENT;
        for (int i = 0; i < indent_level; ++i) {
            INDENT += TAB;
        }
        std::string EQ = std::string(" : ");
        std::string COMMA = std::string(" , ");

        json += INDENT + std::string("{ ") + QT + std::string("RecordSet") +
                QT + std::string(": [") + NL;
        INDENT += TAB;

        for (size_t i = 0; i < m_fields.size(); i++) {
            json += m_fields[i].json(indent_level + 1);
            if (i != m_fields.size() - 1) {
                json += COMMA;
            }
            json += NL;
        }
        INDENT = std::string("");
        for (size_t i = 0; i < static_cast<size_t>(indent_level); ++i) {
            INDENT += TAB;
        }
        json += INDENT + std::string("] }");
        return json;
    }
};

}  // namespace Neon
