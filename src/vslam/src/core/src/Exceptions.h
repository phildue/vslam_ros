//
// Created by phil on 06.07.21.
//

#ifndef DIRECT_IMAGE_ALIGNMENT_EXCEPTIONS_H
#define DIRECT_IMAGE_ALIGNMENT_EXCEPTIONS_H

#include <string>
#include <stdexcept>
namespace pd {

  class Exception: public std::runtime_error
  {
public:
    Exception(const std::string & msg) : std::runtime_error(msg) {
    }
  };
}
#endif //DIRECT_IMAGE_ALIGNMENT_EXCEPTIONS_H
