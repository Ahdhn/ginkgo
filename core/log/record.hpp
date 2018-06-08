/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_LOG_RECORD_HPP_
#define GKO_CORE_LOG_RECORD_HPP_


#include "core/log/logger.hpp"
#include "core/matrix/dense.hpp"


#include <deque>
#include <memory>


namespace gko {
namespace log {


/**
 * Record is a Logger which logs every event to an object. The object can
 * then be accessed at any time by asking the logger to return it.
 */
class Record : public EnablePolymorphicObject<Record, Logger>,
               public EnableCreateMethod<Record> {
    friend class EnablePolymorphicObject<Record, Logger>;
    friend class EnableCreateMethod<Record>;

public:
    using EnablePolymorphicObject<Record, Logger>::EnablePolymorphicObject;

    /**
     * Struct storing the actually logged data
     */
    struct logged_data {
        std::deque<std::string> applies{};
        size_type num_iterations{};
        size_type converged_at_iteration{};
        std::deque<std::unique_ptr<const LinOp>> residuals{};
    };


    void on_iteration_complete(const size_type &num_iterations) const override;

    void on_apply(const std::string &name) const override;

    void on_converged(const size_type &at_iteration,
                      const LinOp *residual) const override;


    /**
     * Returns the logged data
     *
     * @return the logged data
     */
    const logged_data &get() const noexcept { return data_; }

    /**
     * @copydoc ::get()
     */
    logged_data &get() noexcept { return data_; }


protected:
    explicit Record(std::shared_ptr<const gko::Executor> exec,
                    const mask_type &enabled_events, size_type max_storage)
        : EnablePolymorphicObject<Record, Logger>(exec, enabled_events),
          max_storage_{max_storage}
    {}

    /** TODO: Help me with this, really can't know how to do it properly,
     * otherwise clear_impl complains!
     */
    Record &operator=(const Record &other) { return *this; }

    Record &operator=(Record &other) { return *this; }


    mutable logged_data data_{};
    size_type max_storage_{};
};


}  // namespace log
}  // namespace gko


#endif  // GKO_CORE_LOG_RECORD_HPP_
