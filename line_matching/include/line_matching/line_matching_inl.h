#ifndef LINE_MATCHING_LINE_MATCHING_INL_H_
#define LINE_MATCHING_LINE_MATCHING_INL_H_

#include <set>

#include <glog/logging.h>

namespace line_matching {
// Implements a priority queue with a fixed size, feature not explicitly
// supported by the std::priority_queue.
template <class T>
class FixedSizePriorityQueue {
 public:
   // Args: size:                            Fixed size of the priority queue.
   //
   //       ascending_true_descending_false: True if elements in the queue
   //                                        should be sorted in ascending order
   //                                        false for descending order.
   FixedSizePriorityQueue(size_t size, bool ascending_true_descending_false);
   ~FixedSizePriorityQueue();

   // Pushes a new element in the queue.
   void push(const T& new_el);
   // Pops the element with lowest value if in ascending mode and the element
   // with highest value if in descending mode. If the queue is empty, it does
   // nothing.
   void pop();
   // Returns the element with lowest value if in ascending mode and the element
   // with highest value if in descending mode.
   T front();
   // Clears the queue.
   void clear();
   // Returns the size of the queue.
   unsigned int size();
   // True if the queue is empty.
   bool empty();
 protected:
   // Auxiliary class used to keep the element sorted in the queue.
   class comparisonMode {
    public:
      comparisonMode(const bool& ascending_true_descending_false) {
        ascending_true_descending_false_= ascending_true_descending_false;
      }
      bool operator() (const T& el1, const T& el2) const {
        if (ascending_true_descending_false_) {
          return el1 < el2;
        } else {
          return el2 < el1;
        }
      }
    private:
      bool ascending_true_descending_false_;
   };
 private:
   // Internal queue. Sets in C++ STL are implemented as self-balanced binary
   // search trees, that can be used to create the required double-ended
   // priority queue. Multisets allow to have elements with the same value.
   std::multiset<T, FixedSizePriorityQueue<T>::comparisonMode>* queue_;
   // Max (fixed) size of the internal queue.
   size_t max_size_;
   // True: ascending order, false: descending order.
   bool ascending_true_descending_false_;

   // Print queue (for debug).
   void printQueue();
};

template <class T>
FixedSizePriorityQueue<T>::FixedSizePriorityQueue(
    size_t size, bool ascending_true_descending_false) {
  max_size_ = size;
  // Assign input order mode to internal flag.
  ascending_true_descending_false_ = ascending_true_descending_false;
  // Create internal queue.
  queue_ = new std::multiset<T, comparisonMode>(
      comparisonMode(ascending_true_descending_false_));
}

template <class T>
FixedSizePriorityQueue<T>::~FixedSizePriorityQueue() {
  delete queue_;
}

template <class T>
void FixedSizePriorityQueue<T>::push(const T& new_el) {
  if (queue_->size() >= max_size_) {
    T last_el;
    auto last_el_it = queue_->end();
    last_el_it--;
    last_el = *last_el_it;
    if ((new_el < last_el && ascending_true_descending_false_) ||
        (new_el > last_el && !ascending_true_descending_false_)) {
      // Remove "worst element".
      queue_->erase(last_el_it);
    } else {
      // New element is "worse" than the "worst" element.
      return;
    }
  }
  // Push element in the underlying queue.
  queue_->insert(new_el);
}

template <class T>
void FixedSizePriorityQueue<T>::pop() {
  if (empty()) {
    return;
  }
  queue_->erase(queue_->begin());
}

template <class T>
T FixedSizePriorityQueue<T>::front() {
  return *(queue_->begin());
}

template <class T>
void FixedSizePriorityQueue<T>::clear() {
  queue_->clear();
}

template <class T>
unsigned int FixedSizePriorityQueue<T>::size() {
  return queue_->size();
}

template <class T>
bool FixedSizePriorityQueue<T>::empty() {
  return queue_->empty();
}

template <class T>
void FixedSizePriorityQueue<T>::printQueue() {
  LOG(INFO) << "Queue is:";
  for (const auto& el : *queue_) {
    LOG(INFO) << el;
  }
  LOG(INFO) << "-------";
}

}  // namespace line_matching

#endif  // LINE_MATCHING_LINE_MATCHING_INL_H_
