"""Agent 2 - B2B Decision-Maker Contact Finder.

Receives company records produced by Agent 1 (company research) and finds the
decision-makers at those companies who are relevant to buying, approving, or
partnering on IT services, together with their publicly available professional
contact details.

Agent 2 never fabricates people or contact details: every output field is
either backed by evidence returned by the search provider, verified by a
configured verification provider, or left blank.
"""

__version__ = "1.0.0"